import torch
import numpy as np

import arithmetic

chunk_size_cuda = 10000


def encode_binary(x, f):
    """Bernoulli-source arithmetic-encode a flat {0,1} tensor to the already-open binary
    file handle f, using a single empirical probability shared across all elements (ported
    from HAC's utils/encodings_cuda.py encoder()). Returns the encoded size in bits.
    Must be the last thing written to f: decode_binary reads to EOF for the byte stream."""
    x = x.detach().view(-1).to(torch.float32)
    N = x.shape[0]
    prob_1 = float((x.sum() / N).clamp(1e-6, 1 - 1e-6).item())
    p = torch.full((N,), prob_1, device=x.device)
    cdf = torch.stack([torch.zeros_like(p), 1 - p, torch.ones_like(p)], dim=-1)
    sym = x.to(torch.int16).contiguous()
    byte_stream, cnt = arithmetic.arithmetic_encode(sym, cdf, chunk_size_cuda, N, 3)
    cnt_bytes = cnt.cpu().numpy().tobytes()
    stream_bytes = byte_stream.cpu().numpy().tobytes()

    f.write(np.array([prob_1], dtype=np.float32).tobytes())
    f.write(np.array([N, len(cnt_bytes)], dtype=np.int64).tobytes())
    f.write(cnt_bytes)
    f.write(stream_bytes)
    return (len(stream_bytes) + len(cnt_bytes)) * 8 + 32 + 128


def decode_binary(f, device='cuda'):
    """Inverse of encode_binary. f must be positioned at the start of the encoded block;
    reads to EOF, so this can only be the last item in the file."""
    prob_1 = float(np.frombuffer(f.read(4), dtype=np.float32)[0])
    N, len_cnt_bytes = (int(v) for v in np.frombuffer(f.read(16), dtype=np.int64))
    cnt = torch.tensor(np.frombuffer(f.read(len_cnt_bytes), dtype=np.int32).copy(), device=device)
    byte_stream = torch.tensor(np.frombuffer(f.read(), dtype=np.uint8).copy(), device=device)

    p = torch.full((N,), prob_1, device=device)
    cdf = torch.stack([torch.zeros_like(p), 1 - p, torch.ones_like(p)], dim=-1)
    sym = arithmetic.arithmetic_decode(cdf, byte_stream, cnt, chunk_size_cuda, N, 3)
    return sym.to(torch.float32)


def encode_binary_ctx(x, slot_ids, n_slots, f):
    """encode_binary with one empirical probability per slot instead of one for all
    elements. x: flat {0,1} tensor; slot_ids: same-length int tensor in [0, n_slots).
    Unlike encode_binary the block is length-delimited, so several can be written back to
    back and read with decode_binary_ctx. Returns the encoded size in bits."""
    x = x.detach().view(-1).to(torch.float32)
    slot_ids = slot_ids.view(-1)
    N = x.shape[0]
    ones = torch.zeros(n_slots, device=x.device).scatter_add_(0, slot_ids, x)
    total = torch.zeros(n_slots, device=x.device).scatter_add_(0, slot_ids, torch.ones_like(x))
    # Slots that never occur get 0.5 (never used). The clamp only guards log/cdf edge
    # cases: the coder gives every symbol at least 1/65536 of the range regardless.
    table = torch.where(total > 0, ones / total.clamp(min=1), torch.full_like(ones, 0.5)).clamp(1e-6, 1 - 1e-6)
    p = table[slot_ids]
    cdf = torch.stack([torch.zeros_like(p), 1 - p, torch.ones_like(p)], dim=-1)
    sym = x.to(torch.int16).contiguous()
    byte_stream, cnt = arithmetic.arithmetic_encode(sym, cdf, chunk_size_cuda, N, 3)
    cnt_bytes = cnt.cpu().numpy().tobytes()
    stream_bytes = byte_stream.cpu().numpy().tobytes()

    f.write(table.cpu().numpy().astype(np.float32).tobytes())
    f.write(np.array([N, len(cnt_bytes), len(stream_bytes)], dtype=np.int64).tobytes())
    f.write(cnt_bytes)
    f.write(stream_bytes)
    return (len(stream_bytes) + len(cnt_bytes)) * 8 + 32 * n_slots + 192


def decode_binary_ctx(slot_ids, n_slots, f, device='cuda'):
    """Inverse of encode_binary_ctx; leaves f positioned right after this block."""
    table = torch.tensor(np.frombuffer(f.read(4 * n_slots), dtype=np.float32).copy(), device=device)
    N, len_cnt_bytes, len_stream_bytes = (int(v) for v in np.frombuffer(f.read(24), dtype=np.int64))
    assert N == slot_ids.numel(), f"block holds {N} bits but {slot_ids.numel()} slot ids were given"
    cnt = torch.tensor(np.frombuffer(f.read(len_cnt_bytes), dtype=np.int32).copy(), device=device)
    byte_stream = torch.tensor(np.frombuffer(f.read(len_stream_bytes), dtype=np.uint8).copy(), device=device)

    p = table[slot_ids.view(-1)]
    cdf = torch.stack([torch.zeros_like(p), 1 - p, torch.ones_like(p)], dim=-1)
    sym = arithmetic.arithmetic_decode(cdf, byte_stream, cnt, chunk_size_cuda, N, 3)
    return sym.to(torch.float32)


def encode_gaussian(x, mean, scale, Q, file_path):
    """Arithmetic-encode x (1D tensor) under a per-element Gaussian(mean, scale) prior
    quantized with step Q, to file_path. Returns the encoded size in bits.

    The symbol alphabet (round(x/Q) - min_value) must fit in int16, i.e. the chunk's
    quantized-index range must stay under ~32767 -- callers (encode_gaussian_chunk) are
    responsible for keeping chunks within that bound, or values beyond it get silently
    clipped and the round trip stops being exact."""
    assert len(x.shape) == 1
    if not isinstance(Q, torch.Tensor):
        Q = torch.full_like(mean, float(Q))
    Q = Q.view(-1)
    mean = mean.view(-1).contiguous()
    scale = scale.view(-1).contiguous()

    x_int = torch.round(x.view(-1) / Q)
    # calculate_cdf's pybind signature wants arg3/arg4 as plain ints, and silently accepts a
    # 0-dim tensor in the normal case (auto-converted), but throws TypeError instead of
    # clamping when the value itself overflows what an int can hold -- seen in practice as
    # min/max of ~4e30 from a single stray element (a training-instability outlier in x, not
    # necessarily a collapsed Q), which blew past int64 and crashed conduct_encoding at the
    # very last iteration after a 10+ hour run. x_int.min()/.max() themselves were never
    # bounded before this: the *symbols* get clamped to the int16 cap a few lines down, but
    # that clamp is computed FROM min_value/max_value, so it does nothing to protect the
    # bounds calculation itself. Clip around the chunk's median (robust to a handful of wild
    # outliers, unlike min/max) so a bad element degrades to a clipped symbol -- as the
    # SAFE_RANGE warning above already claims happens -- instead of taking down the run.
    median = x_int.median()
    min_value = torch.clamp(x_int.min(), min=median - SAFE_RANGE, max=median)
    max_value = torch.clamp(x_int.max(), min=median, max=median + SAFE_RANGE)
    x_int = torch.clamp(x_int, min=min_value, max=max_value)

    cdf = arithmetic.calculate_cdf(mean, scale, Q, min_value, max_value)
    sym = (x_int - min_value).clamp(min=0, max=32767).to(torch.int16).contiguous()
    byte_stream, cnt = arithmetic.arithmetic_encode(
        sym, cdf, chunk_size_cuda, int(cdf.shape[0]), int(cdf.shape[1])
    )
    cnt_bytes = cnt.cpu().numpy().tobytes()
    stream_bytes = byte_stream.cpu().numpy().tobytes()

    with open(file_path, 'wb') as f:
        f.write(min_value.to(torch.float32).cpu().numpy().tobytes())
        f.write(max_value.to(torch.float32).cpu().numpy().tobytes())
        f.write(np.array([len(cnt_bytes)], dtype=np.int32).tobytes())
        f.write(cnt_bytes)
        f.write(stream_bytes)

    return (len(stream_bytes) + len(cnt_bytes)) * 8 + 32 * 3


def decode_gaussian(mean, scale, Q, file_path):
    """Inverse of encode_gaussian: mean/scale/Q must be reproduced identically on the
    decode side (same model, same anchor positions) for exact reconstruction."""
    if not isinstance(Q, torch.Tensor):
        Q = torch.full_like(mean, float(Q))
    Q = Q.view(-1)
    mean = mean.view(-1).contiguous()
    scale = scale.view(-1).contiguous()

    with open(file_path, 'rb') as f:
        min_value = torch.tensor(np.frombuffer(f.read(4), dtype=np.float32).copy(), device=mean.device)
        max_value = torch.tensor(np.frombuffer(f.read(4), dtype=np.float32).copy(), device=mean.device)
        len_cnt_bytes = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        cnt = torch.tensor(np.frombuffer(f.read(len_cnt_bytes), dtype=np.int32).copy(), device=mean.device)
        byte_stream = torch.tensor(np.frombuffer(f.read(), dtype=np.uint8).copy(), device=mean.device)

    cdf = arithmetic.calculate_cdf(mean, scale, Q, min_value, max_value)
    sym = arithmetic.arithmetic_decode(
        cdf, byte_stream, cnt, chunk_size_cuda, int(cdf.shape[0]), int(cdf.shape[1])
    ).to(mean.device).to(torch.float32)

    return (sym + min_value) * Q


# calculate_cdf allocates an [chunk_len, range] float32 table, and encode_gaussian's
# symbol alphabet is capped at int16 (~32767 distinct values), so a chunk is only safe to
# encode losslessly if its element count AND its round(x/Q) range both stay bounded. Q is
# adaptive per element, so a naive arbitrary-order chunk's range can blow up (a single
# very small Q shifts its element's index a lot) even though x itself is unremarkable.
# Sorting by mean/Q first fixes this in practice: mean/Q is derivable from anchor position
# alone (so the decoder can reproduce the same order without knowing x), and approximates
# round(x/Q) closely for a well-trained model (x should cluster near its predicted mean),
# so it keeps elements with similar quantized indices adjacent -- keeping fixed-size
# chunks' actual ranges well under the cap in practice.
#
# HAC/HAC++ (github.com/YihangChen-ee/HAC-plus, scene/gaussian_model.py conduct_encoding)
# use chunk_size=100_000 for scaling/offsets and 500_000 for feat (split 5-way by channel
# group first), which at first looked like a safe, drop-in fix for this port's much smaller
# (and much more file-count-bloated) 5_000 default. It ISN'T a safe drop-in: HAC filters
# offsets down to `offsets[mask]` (only anchors' actually-active offset slots) before
# chunking, which keeps that tensor's mean/Q distribution narrow; this port's
# conduct_encoding never applies that mask, so its raw `offsets` tensor is far more
# heterogeneous. Tried at chunk_size=100_000 in production, 176 of 177 offsets chunks for
# one scene blew past SAFE_RANGE -- and the resulting min/max values (not just the
# per-symbol values) were never actually bounded anywhere in this file despite the comment
# below claiming they get "lossily clipped", so calculate_cdf crashed on an unrepresentable
# float instead of degrading gracefully (see encode_gaussian's clamp, added after this was
# caught by a 10-hour run dying at the very last iteration). That crash is now fixed for any
# chunk_size, but 100_000 still means near-every offsets chunk gets silently, heavily
# lossy-clipped -- a real quality regression, not just a file-count tradeoff. Back to 5_000,
# which never triggered this warning across many prior runs.
DEFAULT_CHUNK_SIZE = 5_000
SAFE_RANGE = 20_000  # comfortably under the int16 (32767) symbol cap


def fit_gmm_1d_2comp(x, n_iter=50):
    """Fits a 2-component 1D Gaussian mixture to x via plain-torch EM. For small
    side-info-only global priors (e.g. the per-checkpoint anchor-mask value distribution)
    where a per-element hyper-network prediction would be overkill -- the decoder only
    needs the 6 resulting scalars (2 means, 2 scales, 2 mixture weights) saved to the
    bitstream, not a per-anchor prediction. The encoder's convergence doesn't need to be
    reproducible: whatever it converges to gets saved, and decode just reuses those exact
    saved numbers, so there's no encode/decode-side determinism requirement here (unlike
    the rest of this module, where mean/scale must be IDENTICALLY recomputed on both
    sides)."""
    x = x.view(-1).float()
    N = x.numel()
    init_idx = torch.randperm(N, device=x.device)[:2]
    means = x[init_idx].clone()
    scales = x.std().clamp(min=1e-3).expand(2).clone()
    weights = torch.full((2,), 0.5, device=x.device)
    for _ in range(n_iter):
        log_w = torch.log(weights.clamp(min=1e-12))
        d0 = torch.distributions.Normal(means[0], scales[0].clamp(min=1e-6)).log_prob(x) + log_w[0]
        d1 = torch.distributions.Normal(means[1], scales[1].clamp(min=1e-6)).log_prob(x) + log_w[1]
        logsum = torch.logaddexp(d0, d1)
        r0 = torch.exp(d0 - logsum)
        r1 = torch.exp(d1 - logsum)
        N0, N1 = r0.sum().clamp(min=1e-6), r1.sum().clamp(min=1e-6)
        means = torch.stack([(r0 * x).sum() / N0, (r1 * x).sum() / N1])
        scales = torch.stack([
            ((r0 * (x - means[0]) ** 2).sum() / N0).sqrt().clamp(min=1e-3),
            ((r1 * (x - means[1]) ** 2).sum() / N1).sqrt().clamp(min=1e-3),
        ])
        weights = torch.stack([N0, N1]) / N
    return means.detach(), scales.detach(), weights.detach()


def _locality_sort_key(mean, Q):
    return mean / Q


def encode_gaussian_chunk(x, mean, scale, Q, file_path, chunk_size=DEFAULT_CHUNK_SIZE):
    x, mean, scale = x.view(-1), mean.view(-1), scale.view(-1)
    Q = Q.view(-1) if isinstance(Q, torch.Tensor) else torch.full_like(mean, float(Q))
    N = x.shape[0]

    sort_idx = torch.argsort(_locality_sort_key(mean, Q))
    x, mean, scale, Q = x[sort_idx], mean[sort_idx], scale[sort_idx], Q[sort_idx]

    n_chunks = int(np.ceil(N / chunk_size))
    pad = n_chunks * chunk_size - N
    if pad > 0:
        x_int_padded = torch.cat([torch.round(x / Q), x.new_zeros(pad)]).view(n_chunks, chunk_size)
    else:
        x_int_padded = torch.round(x / Q).view(n_chunks, chunk_size)
    ranges = x_int_padded.max(dim=1).values - x_int_padded.min(dim=1).values
    n_unsafe = int((ranges[:n_chunks - (1 if pad > 0 else 0)] > SAFE_RANGE).sum().item())
    if n_unsafe > 0:
        print(f"[arithmetic_coding] WARNING: {n_unsafe}/{n_chunks} chunks in {file_path} exceed "
              f"the safe quantized-index range ({SAFE_RANGE}); those chunks will be lossily clipped. "
              f"Consider a smaller chunk_size or a tighter adaptive-Q floor.")

    total_bits = 0
    for c in range(n_chunks):
        s, e = c * chunk_size, min((c + 1) * chunk_size, N)
        total_bits += encode_gaussian(x[s:e], mean[s:e], scale[s:e], Q[s:e],
                                       file_path.replace('.b', f'_{c}.b'))
    return total_bits


def decode_gaussian_chunk(mean, scale, Q, file_path, chunk_size=DEFAULT_CHUNK_SIZE):
    mean, scale = mean.view(-1), scale.view(-1)
    Q = Q.view(-1) if isinstance(Q, torch.Tensor) else torch.full_like(mean, float(Q))
    N = mean.shape[0]
    sort_idx = torch.argsort(_locality_sort_key(mean, Q))  # must match the encoder exactly
    mean_s, scale_s, Q_s = mean[sort_idx], scale[sort_idx], Q[sort_idx]

    n_chunks = int(np.ceil(N / chunk_size))
    out_sorted = []
    for c in range(n_chunks):
        s, e = c * chunk_size, min((c + 1) * chunk_size, N)
        out_sorted.append(decode_gaussian(mean_s[s:e], scale_s[s:e], Q_s[s:e], file_path.replace('.b', f'_{c}.b')))
    out_sorted = torch.cat(out_sorted, dim=0)

    out = torch.empty_like(out_sorted)
    out[sort_idx] = out_sorted  # undo the sort to restore original element order
    return out


def encode_gaussian_mixed(x, mean1, mean2, scale1, scale2, probs1, probs2, Q, file_path):
    """encode_gaussian's 2-component-mixture counterpart (see
    utils/entropy_models.py Entropy_gaussian_mix_prob_2), for the intra-anchor channel
    context model: component 1 is the hash-grid (inter-anchor) Gaussian, component 2 is
    the intra-anchor MLP's adjustment. Mixture CDF = probs1*CDF1 + probs2*CDF2 is exact
    for any two-component mixture, so this is calculate_cdf called twice and linearly
    combined -- no new CUDA code needed, and the crash-safety clamp (see encode_gaussian)
    applies identically since it only bounds x_int/min_value/max_value, not the CDF
    itself."""
    assert len(x.shape) == 1
    if not isinstance(Q, torch.Tensor):
        Q = torch.full_like(mean1, float(Q))
    Q = Q.view(-1)
    mean1, mean2 = mean1.view(-1).contiguous(), mean2.view(-1).contiguous()
    scale1, scale2 = scale1.view(-1).contiguous(), scale2.view(-1).contiguous()
    probs1, probs2 = probs1.view(-1).contiguous(), probs2.view(-1).contiguous()

    x_int = torch.round(x.view(-1) / Q)
    median = x_int.median()
    min_value = torch.clamp(x_int.min(), min=median - SAFE_RANGE, max=median)
    max_value = torch.clamp(x_int.max(), min=median, max=median + SAFE_RANGE)
    x_int = torch.clamp(x_int, min=min_value, max=max_value)

    cdf1 = arithmetic.calculate_cdf(mean1, scale1, Q, min_value, max_value)
    cdf2 = arithmetic.calculate_cdf(mean2, scale2, Q, min_value, max_value)
    cdf = probs1.unsqueeze(-1) * cdf1 + probs2.unsqueeze(-1) * cdf2
    sym = (x_int - min_value).clamp(min=0, max=32767).to(torch.int16).contiguous()
    byte_stream, cnt = arithmetic.arithmetic_encode(
        sym, cdf, chunk_size_cuda, int(cdf.shape[0]), int(cdf.shape[1])
    )
    cnt_bytes = cnt.cpu().numpy().tobytes()
    stream_bytes = byte_stream.cpu().numpy().tobytes()

    with open(file_path, 'wb') as f:
        f.write(min_value.to(torch.float32).cpu().numpy().tobytes())
        f.write(max_value.to(torch.float32).cpu().numpy().tobytes())
        f.write(np.array([len(cnt_bytes)], dtype=np.int32).tobytes())
        f.write(cnt_bytes)
        f.write(stream_bytes)

    return (len(stream_bytes) + len(cnt_bytes)) * 8 + 32 * 3


def decode_gaussian_mixed(mean1, mean2, scale1, scale2, probs1, probs2, Q, file_path):
    """Inverse of encode_gaussian_mixed."""
    if not isinstance(Q, torch.Tensor):
        Q = torch.full_like(mean1, float(Q))
    Q = Q.view(-1)
    mean1, mean2 = mean1.view(-1).contiguous(), mean2.view(-1).contiguous()
    scale1, scale2 = scale1.view(-1).contiguous(), scale2.view(-1).contiguous()
    probs1, probs2 = probs1.view(-1).contiguous(), probs2.view(-1).contiguous()

    with open(file_path, 'rb') as f:
        min_value = torch.tensor(np.frombuffer(f.read(4), dtype=np.float32).copy(), device=mean1.device)
        max_value = torch.tensor(np.frombuffer(f.read(4), dtype=np.float32).copy(), device=mean1.device)
        len_cnt_bytes = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        cnt = torch.tensor(np.frombuffer(f.read(len_cnt_bytes), dtype=np.int32).copy(), device=mean1.device)
        byte_stream = torch.tensor(np.frombuffer(f.read(), dtype=np.uint8).copy(), device=mean1.device)

    cdf1 = arithmetic.calculate_cdf(mean1, scale1, Q, min_value, max_value)
    cdf2 = arithmetic.calculate_cdf(mean2, scale2, Q, min_value, max_value)
    cdf = probs1.unsqueeze(-1) * cdf1 + probs2.unsqueeze(-1) * cdf2
    sym = arithmetic.arithmetic_decode(
        cdf, byte_stream, cnt, chunk_size_cuda, int(cdf.shape[0]), int(cdf.shape[1])
    ).to(mean1.device).to(torch.float32)

    return (sym + min_value) * Q


def encode_gaussian_mixed_chunk(x, mean1, mean2, scale1, scale2, probs1, probs2, Q, file_path,
                                 chunk_size=DEFAULT_CHUNK_SIZE):
    x, mean1, mean2 = x.view(-1), mean1.view(-1), mean2.view(-1)
    scale1, scale2 = scale1.view(-1), scale2.view(-1)
    probs1, probs2 = probs1.view(-1), probs2.view(-1)
    Q = Q.view(-1) if isinstance(Q, torch.Tensor) else torch.full_like(mean1, float(Q))
    N = x.shape[0]

    sort_idx = torch.argsort(_locality_sort_key(mean1, Q))
    x, mean1, mean2, scale1, scale2, probs1, probs2, Q = (
        x[sort_idx], mean1[sort_idx], mean2[sort_idx], scale1[sort_idx], scale2[sort_idx],
        probs1[sort_idx], probs2[sort_idx], Q[sort_idx])

    n_chunks = int(np.ceil(N / chunk_size))
    total_bits = 0
    for c in range(n_chunks):
        s, e = c * chunk_size, min((c + 1) * chunk_size, N)
        total_bits += encode_gaussian_mixed(
            x[s:e], mean1[s:e], mean2[s:e], scale1[s:e], scale2[s:e], probs1[s:e], probs2[s:e], Q[s:e],
            file_path.replace('.b', f'_{c}.b'))
    return total_bits


def decode_gaussian_mixed_chunk(mean1, mean2, scale1, scale2, probs1, probs2, Q, file_path,
                                 chunk_size=DEFAULT_CHUNK_SIZE):
    mean1, mean2 = mean1.view(-1), mean2.view(-1)
    scale1, scale2 = scale1.view(-1), scale2.view(-1)
    probs1, probs2 = probs1.view(-1), probs2.view(-1)
    Q = Q.view(-1) if isinstance(Q, torch.Tensor) else torch.full_like(mean1, float(Q))
    N = mean1.shape[0]
    sort_idx = torch.argsort(_locality_sort_key(mean1, Q))  # must match the encoder exactly
    mean1_s, mean2_s = mean1[sort_idx], mean2[sort_idx]
    scale1_s, scale2_s = scale1[sort_idx], scale2[sort_idx]
    probs1_s, probs2_s = probs1[sort_idx], probs2[sort_idx]
    Q_s = Q[sort_idx]

    n_chunks = int(np.ceil(N / chunk_size))
    out_sorted = []
    for c in range(n_chunks):
        s, e = c * chunk_size, min((c + 1) * chunk_size, N)
        out_sorted.append(decode_gaussian_mixed(
            mean1_s[s:e], mean2_s[s:e], scale1_s[s:e], scale2_s[s:e], probs1_s[s:e], probs2_s[s:e], Q_s[s:e],
            file_path.replace('.b', f'_{c}.b')))
    out_sorted = torch.cat(out_sorted, dim=0)

    out = torch.empty_like(out_sorted)
    out[sort_idx] = out_sorted
    return out
