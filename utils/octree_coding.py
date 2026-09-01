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
collected into one flat {0,1} array and arithmetic-coded with a single empirical
probability (utils/arithmetic_coding.py: encode_binary/decode_binary) -- the same
mechanism used for the hash grid. Most of the compression comes from deep levels being
overwhelmingly "not occupied" (sparse point set relative to full grid resolution).
"""

import torch
import numpy as np

from utils.arithmetic_coding import encode_binary, decode_binary


def _axis_bits_for(int_coords):
    # int_coords: [N] non-negative int64. Bits needed to represent 0..max exactly.
    max_val = int(int_coords.max().item()) if int_coords.numel() > 0 else 0
    return max(1, int(max_val).bit_length())


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
    point_node = torch.zeros(N, dtype=torch.int64, device=device)

    all_bits = []
    for level in range(max_bits):
        active_axes = [a for a in range(D) if axis_bits[a] > level]
        if not active_axes:
            break
        k = len(active_axes)
        num_nodes = active_prefixes.shape[0]

        child_bits = torch.zeros(N, dtype=torch.int64, device=device)
        for i, a in enumerate(active_axes):
            bit_pos = axis_bits[a] - 1 - level
            bit = (anchor_int[:, a] >> bit_pos) & 1
            child_bits |= (bit << i)

        occ = torch.zeros(num_nodes, 1 << k, dtype=torch.float32, device=device)
        occ[point_node, child_bits] = 1.0
        all_bits.append(occ.view(-1))

        # New active node set = occupied (parent, child) pairs, in a fixed traversal order
        # (row-major nonzero) that decode_positions replays identically.
        node_idx, child_val = (occ > 0.5).nonzero(as_tuple=True)
        new_prefixes = active_prefixes[node_idx].clone()
        for i, a in enumerate(active_axes):
            bit = (child_val >> i) & 1
            new_prefixes[:, a] = (new_prefixes[:, a] << 1) | bit
        active_prefixes = new_prefixes

        lookup = torch.full((num_nodes, 1 << k), -1, dtype=torch.int64, device=device)
        lookup[node_idx, child_val] = torch.arange(node_idx.shape[0], device=device)
        point_node = lookup[point_node, child_bits]

    occ_bits = torch.cat(all_bits) if all_bits else torch.zeros(0, device=device)

    with open(path, 'wb') as f:
        f.write(np.array([N, D], dtype=np.int64).tobytes())
        f.write(np.array(axis_bits, dtype=np.int64).tobytes())
        encode_binary(occ_bits, f)

    return point_node  # anchor_int[i] belongs at row point_node[i] of the decoded tensor


def decode_positions(path, device='cuda'):
    """Inverse of encode_positions. Returns [N, D] int64 grid coordinates in the tree's
    canonical order (see encode_positions' docstring about reordering attributes)."""
    with open(path, 'rb') as f:
        N, D = (int(v) for v in np.frombuffer(f.read(16), dtype=np.int64))
        axis_bits = [int(v) for v in np.frombuffer(f.read(8 * D), dtype=np.int64)]
        occ_bits = decode_binary(f, device=device)

    max_bits = max(axis_bits)
    active_prefixes = torch.zeros(1, D, dtype=torch.int64, device=device)
    offset = 0
    for level in range(max_bits):
        active_axes = [a for a in range(D) if axis_bits[a] > level]
        if not active_axes:
            break
        k = len(active_axes)
        num_nodes = active_prefixes.shape[0]
        n_bits = num_nodes * (1 << k)
        level_bits = occ_bits[offset:offset + n_bits].view(num_nodes, 1 << k)
        offset += n_bits

        node_idx, child_val = (level_bits > 0.5).nonzero(as_tuple=True)
        new_prefixes = active_prefixes[node_idx].clone()
        for i, a in enumerate(active_axes):
            bit = (child_val >> i) & 1
            new_prefixes[:, a] = (new_prefixes[:, a] << 1) | bit
        active_prefixes = new_prefixes

    assert active_prefixes.shape[0] == N, \
        f"decoded {active_prefixes.shape[0]} leaves, expected {N} -- corrupt bitstream?"
    return active_prefixes
