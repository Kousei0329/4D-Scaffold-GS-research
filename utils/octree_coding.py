"""Lossless entropy coding of anchor positions (x, y, z, t) via an anisotropic 4D
hyperoctree (a "2^k-ary tree" generalizing the familiar 3D octree to 4 axes).

Anchors already live on a regular grid (voxel_size for x/y/z, t_grid_size for t), so each
axis needs only a fixed number of bits to represent exactly. Space and time need very
different numbers of bits (a fine spatial voxel_size over a large scene vs. a coarse
t_grid_size over a short duration), so a naive 16-way (2^4) split at every level would
either waste bits over-resolving whichever axis runs out first, or (worse) simply run out
of bits for it while the others still need more depth. Instead, this tree splits along
whichever axes still have bits remaining at each level -- 16-way while all 4 axes are still
active, dropping to 8-way (an ordinary octree) once time is fully resolved, matching how
GridEncoder's axis-triplet decomposition works around the same space/time resolution
mismatch elsewhere in this codebase.

At each level, the occupancy pattern of every currently-occupied node's children is
collected into one flat {0,1} array and arithmetic-coded as its own stream
(utils/arithmetic_coding.py: encode_binary_ctx), with one empirical probability either per
child slot alone, or per (this node's slot, the PARENT node's own slot one level up) pair.
The latter captures a real extra order of correlation (a node that's itself in a
densely-occupied region of the tree tends to have differently-shaped children than one in
a sparse region) for free, since the parent's slot was already decoded at the previous
level on both the encode and decode side -- but its probability table costs 4 bytes per
(parent slot, child slot) pair, which can exceed what it saves when a level has few active
nodes (measured +45% WORSE on coffee_martini's ~4k anchors if applied unconditionally: its
early levels never have enough samples per pair to amortize the table). So at every level,
both options' bit costs are estimated (_estimate_bits) from the same data each would
actually encode, the cheaper one is used, and which one was picked is written as a 1-byte
flag ahead of that level's block -- never worse than always using the plain per-slot
context, on any scene, by more than the estimate's small mismatch against the arithmetic
coder's real output. The parent-slot context is unconditionally unavailable at level 0 (the
root has no parent).
"""

import torch
import numpy as np

from utils.arithmetic_coding import encode_binary_ctx, decode_binary_ctx


def _axis_bits_for(int_coords):
    # int_coords: [N] non-negative int64. Bits needed to represent 0..max exactly.
    max_val = int(int_coords.max().item()) if int_coords.numel() > 0 else 0
    return max(1, int(max_val).bit_length())


def _estimate_bits(x, ctx_ids, n_ctx):
    """Analytic estimate of what encode_binary_ctx(x, ctx_ids, n_ctx, ...) would cost:
    each context's empirical probability (same fit encode_binary_ctx itself uses) times the
    cross-entropy of x under it, plus the probability table's own fixed overhead. Doesn't
    touch disk -- used only to pick between candidate context schemes before committing to
    one. Closely tracks the real arithmetic-coded size (near-optimal for the given
    probabilities), which is all picking the cheaper OPTION needs; it need not be exact."""
    ones = torch.zeros(n_ctx, device=x.device).scatter_add_(0, ctx_ids, x)
    total = torch.zeros(n_ctx, device=x.device).scatter_add_(0, ctx_ids, torch.ones_like(x))
    p = torch.where(total > 0, ones / total.clamp(min=1), torch.full_like(ones, 0.5)).clamp(1e-6, 1 - 1e-6)
    p_elem = p[ctx_ids]
    cross_entropy_bits = -(x * torch.log2(p_elem) + (1 - x) * torch.log2(1 - p_elem)).sum().item()
    return cross_entropy_bits + 32 * n_ctx + 192


def _level_contexts(num_nodes, n_slots, active_parent_slot, parent_n_slots, device):
    """Both candidate (ctx_ids, n_ctx) schemes for one level: plain per-slot, and
    per-(parent slot, child slot). Shared by encode and decode so the two can never
    disagree about what each scheme concretely is."""
    slot_ids = torch.arange(n_slots, device=device).repeat(num_nodes)
    flat = (slot_ids, n_slots)
    parent = (active_parent_slot.repeat_interleave(n_slots) * n_slots + slot_ids, parent_n_slots * n_slots)
    return flat, parent


def encode_positions(anchor_int, path):
    """anchor_int: [N, D] non-negative int64 grid coordinates (already shifted so every
    entry is >= 0). Writes the geometry bitstream to path. Returns a permutation `perm`
    such that anchor_int[perm] is in the tree's natural (canonical) traversal order --
    callers should reorder any per-anchor attributes by this same perm before encoding
    them, so that decode_positions' output order lines up with the rest of the decoded
    attributes with no extra bookkeeping."""
    N, D = anchor_int.shape
    device = anchor_int.device
    axis_bits = [_axis_bits_for(anchor_int[:, a]) for a in range(D)]
    max_bits = max(axis_bits)

    active_prefixes = torch.zeros(1, D, dtype=torch.int64, device=device)
    active_parent_slot = torch.zeros(1, dtype=torch.int64, device=device)  # no parent yet
    parent_n_slots = 1
    point_node = torch.zeros(N, dtype=torch.int64, device=device)

    level_blocks = []  # (occupancy bits, context id of each bit, number of context ids, used_parent_ctx)
    for level in range(max_bits):
        active_axes = [a for a in range(D) if axis_bits[a] > level]
        if not active_axes:
            break
        k = len(active_axes)
        n_slots = 1 << k
        num_nodes = active_prefixes.shape[0]

        child_bits = torch.zeros(N, dtype=torch.int64, device=device)
        for i, a in enumerate(active_axes):
            bit_pos = axis_bits[a] - 1 - level
            bit = (anchor_int[:, a] >> bit_pos) & 1
            child_bits |= (bit << i)

        occ = torch.zeros(num_nodes, n_slots, dtype=torch.float32, device=device)
        occ[point_node, child_bits] = 1.0
        bits_flat_tensor = occ.view(-1)

        (flat_ids, flat_n), (parent_ids, parent_n) = _level_contexts(
            num_nodes, n_slots, active_parent_slot, parent_n_slots, device)
        cost_flat = _estimate_bits(bits_flat_tensor, flat_ids, flat_n)
        cost_parent = _estimate_bits(bits_flat_tensor, parent_ids, parent_n)
        use_parent_ctx = cost_parent < cost_flat
        ctx_ids, n_ctx = (parent_ids, parent_n) if use_parent_ctx else (flat_ids, flat_n)
        level_blocks.append((bits_flat_tensor, ctx_ids, n_ctx, use_parent_ctx))

        # New active node set = occupied (parent, child) pairs, in a fixed traversal order
        # (row-major nonzero) that decode_positions replays identically.
        node_idx, child_val = (occ > 0.5).nonzero(as_tuple=True)
        new_prefixes = active_prefixes[node_idx].clone()
        for i, a in enumerate(active_axes):
            bit = (child_val >> i) & 1
            new_prefixes[:, a] = (new_prefixes[:, a] << 1) | bit
        active_prefixes = new_prefixes
        active_parent_slot = child_val
        parent_n_slots = n_slots

        lookup = torch.full((num_nodes, n_slots), -1, dtype=torch.int64, device=device)
        lookup[node_idx, child_val] = torch.arange(node_idx.shape[0], device=device)
        point_node = lookup[point_node, child_bits]

    with open(path, 'wb') as f:
        f.write(np.array([N, D], dtype=np.int64).tobytes())
        f.write(np.array(axis_bits, dtype=np.int64).tobytes())
        for bits, ctx_ids, n_ctx, used_parent_ctx in level_blocks:
            f.write(np.array([1 if used_parent_ctx else 0], dtype=np.uint8).tobytes())
            encode_binary_ctx(bits, ctx_ids, n_ctx, f)

    return point_node  # anchor_int[i] belongs at row point_node[i] of the decoded tensor


def decode_positions(path, device='cuda'):
    """Inverse of encode_positions. Returns [N, D] int64 grid coordinates in the tree's
    canonical order (see encode_positions' docstring about reordering attributes)."""
    with open(path, 'rb') as f:
        N, D = (int(v) for v in np.frombuffer(f.read(16), dtype=np.int64))
        axis_bits = [int(v) for v in np.frombuffer(f.read(8 * D), dtype=np.int64)]

        max_bits = max(axis_bits)
        active_prefixes = torch.zeros(1, D, dtype=torch.int64, device=device)
        active_parent_slot = torch.zeros(1, dtype=torch.int64, device=device)
        parent_n_slots = 1
        for level in range(max_bits):
            active_axes = [a for a in range(D) if axis_bits[a] > level]
            if not active_axes:
                break
            k = len(active_axes)
            n_slots = 1 << k
            num_nodes = active_prefixes.shape[0]

            used_parent_ctx = bool(np.frombuffer(f.read(1), dtype=np.uint8)[0])
            (flat_ids, flat_n), (parent_ids, parent_n) = _level_contexts(
                num_nodes, n_slots, active_parent_slot, parent_n_slots, device)
            ctx_ids, n_ctx = (parent_ids, parent_n) if used_parent_ctx else (flat_ids, flat_n)
            level_bits = decode_binary_ctx(ctx_ids, n_ctx, f, device=device).view(num_nodes, n_slots)

            node_idx, child_val = (level_bits > 0.5).nonzero(as_tuple=True)
            new_prefixes = active_prefixes[node_idx].clone()
            for i, a in enumerate(active_axes):
                bit = (child_val >> i) & 1
                new_prefixes[:, a] = (new_prefixes[:, a] << 1) | bit
            active_prefixes = new_prefixes
            active_parent_slot = child_val
            parent_n_slots = n_slots

    assert active_prefixes.shape[0] == N, \
        f"decoded {active_prefixes.shape[0]} leaves, expected {N} -- corrupt bitstream?"
    return active_prefixes
