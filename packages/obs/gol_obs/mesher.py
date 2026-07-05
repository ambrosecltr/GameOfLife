"""Voxel chunk -> triangle mesh, vectorized.

Exposed-face extraction: a face is emitted where a non-air block borders air.
No greedy merging — chunks are small (16x16 columns) and only dirty chunks are
re-meshed, so simple wins. Faces are shaded per direction so the terrain reads
in 3D without scene lighting.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from gol_world.blocks import COLOR, Block
from gol_world.grid import CHUNK

# Per-direction data: (offset, four CCW-from-outside corners, shade).
_FACES: list[tuple[tuple[int, int, int], npt.NDArray[np.float32], float]] = [
    (
        (1, 0, 0),
        np.array([(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)], dtype=np.float32),
        0.80,
    ),
    (
        (-1, 0, 0),
        np.array([(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)], dtype=np.float32),
        0.80,
    ),
    (
        (0, 1, 0),
        np.array([(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)], dtype=np.float32),
        0.70,
    ),
    (
        (0, -1, 0),
        np.array([(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], dtype=np.float32),
        0.70,
    ),
    (
        (0, 0, 1),
        np.array([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], dtype=np.float32),
        1.00,
    ),
    (
        (0, 0, -1),
        np.array([(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)], dtype=np.float32),
        0.45,
    ),
]

ChunkMesh = tuple[
    npt.NDArray[np.float32],  # vertices (V, 3)
    npt.NDArray[np.uint32],  # triangles (T, 3)
    npt.NDArray[np.uint8],  # vertex colors (V, 3)
]


def chunk_mesh(blocks: npt.NDArray[np.uint8], chunk_x: int, chunk_y: int) -> ChunkMesh | None:
    """Mesh one 16x16-column chunk of the full block array.

    Returns None for an empty (all-air) chunk.
    """
    sx, sy, sz = blocks.shape
    x0, y0 = chunk_x * CHUNK, chunk_y * CHUNK
    x1, y1 = min(x0 + CHUNK, sx), min(y0 + CHUNK, sy)

    # Pad by one so neighbor lookups at chunk borders see the real world;
    # beyond the world itself, padding is AIR (world edges render as cliffs).
    px0, py0 = max(x0 - 1, 0), max(y0 - 1, 0)
    px1, py1 = min(x1 + 1, sx), min(y1 + 1, sy)
    region = blocks[px0:px1, py0:py1, :]
    pad = (
        (1 - (x0 - px0), 1 - (px1 - x1)),
        (1 - (y0 - py0), 1 - (py1 - y1)),
        (1, 1),
    )
    padded = np.pad(region, pad, constant_values=Block.AIR)
    core = padded[1:-1, 1:-1, 1:-1]

    if not core.any():
        return None

    vert_parts: list[npt.NDArray[np.float32]] = []
    color_parts: list[npt.NDArray[np.uint8]] = []

    for (dx, dy, dz), corners, shade in _FACES:
        neighbor = padded[
            1 + dx : padded.shape[0] - 1 + dx or None,
            1 + dy : padded.shape[1] - 1 + dy or None,
            1 + dz : padded.shape[2] - 1 + dz or None,
        ]
        visible = (core != Block.AIR) & (neighbor == Block.AIR)
        if not visible.any():
            continue
        cx, cy, cz = np.nonzero(visible)
        base = np.stack([cx + x0, cy + y0, cz], axis=1).astype(np.float32)
        quads = base[:, None, :] + corners[None, :, :]  # (N, 4, 3)
        vert_parts.append(quads.reshape(-1, 3))
        shaded = (COLOR[core[cx, cy, cz]].astype(np.float32) * shade).astype(np.uint8)
        color_parts.append(np.repeat(shaded, 4, axis=0))

    if not vert_parts:
        return None

    vertices = np.concatenate(vert_parts)
    colors = np.concatenate(color_parts)
    n_quads = len(vertices) // 4
    idx = np.arange(n_quads, dtype=np.uint32) * 4
    triangles = np.empty((n_quads * 2, 3), dtype=np.uint32)
    triangles[0::2] = np.stack([idx, idx + 1, idx + 2], axis=1)
    triangles[1::2] = np.stack([idx, idx + 2, idx + 3], axis=1)
    return vertices, triangles, colors
