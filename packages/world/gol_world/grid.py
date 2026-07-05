"""Dense voxel grid with chunk-level dirty tracking.

The world is finite and small enough (256x256x64 = 4 MB at uint8) that one
dense array beats any sparse cleverness. Chunks (16x16 columns, full height)
exist only as generation units and as the granularity of renderer updates.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from gol_world.blocks import SOLID, Block

CHUNK = 16

BlockUpdate = tuple[int, int, int, int]  # x, y, z, new block id


class VoxelGrid:
    def __init__(self, blocks: npt.NDArray[np.uint8]) -> None:
        if blocks.ndim != 3 or blocks.dtype != np.uint8:
            raise ValueError(f"expected 3D uint8 array, got {blocks.shape} {blocks.dtype}")
        self.blocks = blocks
        self.dirty_chunks: set[tuple[int, int]] = set()
        self._updates: list[BlockUpdate] = []

    @classmethod
    def empty(cls, size: tuple[int, int, int]) -> VoxelGrid:
        return cls(np.zeros(size, dtype=np.uint8))

    @property
    def size(self) -> tuple[int, int, int]:
        sx, sy, sz = self.blocks.shape
        return (sx, sy, sz)

    def in_bounds(self, x: int, y: int, z: int) -> bool:
        sx, sy, sz = self.blocks.shape
        return bool(0 <= x < sx and 0 <= y < sy and 0 <= z < sz)

    def get_block(self, x: int, y: int, z: int) -> int:
        return int(self.blocks[x, y, z])

    def set_block(self, x: int, y: int, z: int, block: int) -> None:
        """Set a block, tracking the change for renderer and event consumers."""
        if int(self.blocks[x, y, z]) == block:
            return
        self.blocks[x, y, z] = block
        self.dirty_chunks.add((x // CHUNK, y // CHUNK))
        self._updates.append((x, y, z, block))

    def is_solid(self, x: int, y: int, z: int) -> bool:
        """Out-of-bounds counts as solid: the world border is a wall."""
        if not self.in_bounds(x, y, z):
            return True
        return bool(SOLID[self.blocks[x, y, z]])

    def column_height(self, x: int, y: int) -> int:
        """Z of the highest non-air block in the column (-1 if all air)."""
        col = self.blocks[x, y, :]
        nonair = np.nonzero(col != Block.AIR)[0]
        return int(nonair[-1]) if nonair.size else -1

    def consume_updates(self) -> list[BlockUpdate]:
        """Drain block changes accumulated since the last call."""
        updates, self._updates = self._updates, []
        return updates

    def consume_dirty_chunks(self) -> set[tuple[int, int]]:
        """Drain the set of chunks whose meshes need rebuilding."""
        dirty, self.dirty_chunks = self.dirty_chunks, set()
        return dirty
