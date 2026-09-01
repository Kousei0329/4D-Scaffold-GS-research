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
    min_value = x_int.min()
    max_value = x_int.max()

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
DEFAULT_CHUNK_SIZE = 5_000
SAFE_RANGE = 20_000  # comfortably under the int16 (32767) symbol cap


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
