import numpy as np
from gol_obs.mesher import chunk_mesh
from gol_world.blocks import Block
from gol_world.config import WorldConfig
from gol_world.terrain import generate


def test_empty_chunk_is_none() -> None:
    blocks = np.zeros((32, 32, 16), dtype=np.uint8)
    assert chunk_mesh(blocks, 0, 0) is None


def test_single_block_has_six_faces() -> None:
    blocks = np.zeros((16, 16, 16), dtype=np.uint8)
    blocks[5, 5, 5] = Block.ROCK
    mesh = chunk_mesh(blocks, 0, 0)
    assert mesh is not None
    vertices, triangles, colors = mesh
    assert len(vertices) == 6 * 4
    assert len(triangles) == 6 * 2
    assert len(colors) == len(vertices)
    assert triangles.max() == len(vertices) - 1


def test_buried_block_faces_are_culled() -> None:
    blocks = np.zeros((16, 16, 16), dtype=np.uint8)
    blocks[4:7, 4:7, 4:7] = Block.ROCK  # 3x3x3 cube: only outer faces show
    mesh = chunk_mesh(blocks, 0, 0)
    assert mesh is not None
    vertices, _, _ = mesh
    assert len(vertices) == 6 * 9 * 4  # 6 sides x 9 visible faces each


def test_chunk_border_faces_use_real_neighbors() -> None:
    # A slab spanning the chunk border: no wall of faces at x=16.
    blocks = np.zeros((32, 16, 16), dtype=np.uint8)
    blocks[:, :, 0] = Block.ROCK
    left = chunk_mesh(blocks, 0, 0)
    assert left is not None
    vertices, _, _ = left
    # Top + bottom faces plus the outer -x and +/-y rims; none at the x=16 seam.
    xs = vertices[:, 0]
    seam_faces = ((xs == 16.0).reshape(-1, 4).all(axis=1)).sum()
    assert seam_faces == 0


def test_terrain_meshes_everywhere() -> None:
    grid = generate(WorldConfig(seed=1, size=(64, 64, 48)))
    for cx in range(4):
        for cy in range(4):
            assert chunk_mesh(grid.blocks, cx, cy) is not None
