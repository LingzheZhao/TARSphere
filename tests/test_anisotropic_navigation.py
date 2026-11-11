import numpy as np
from tarsphere.harness.perception.backends.gt_bev import OfficialOccupancyBevMap
from tarsphere.harness.planner.planner import Planner, erode_navigable_mask


def make_map(tmp_path, dx=0.1, dy=0.2):
    a = np.ones((22, 22), dtype=np.float32)
    a[0, 1:] = np.arange(21) * dx
    a[1:, 0] = np.arange(21)[::-1] * dy
    a[11, 11] = 0
    path = tmp_path / "occupancy.npy"
    np.save(path, a)
    return OfficialOccupancyBevMap(occupancy_path=path)


def test_metric_clearance_and_axis_roundtrip(tmp_path):
    bev = make_map(tmp_path)
    np.testing.assert_allclose(bev.cell_size_xy_m, (0.1, 0.2), rtol=1e-5)
    for cell in [(5, 7), (15, 16)]:
        assert tuple(bev.world_to_cell([bev.cell_to_world(*cell)])[0]) == cell
    p = Planner(bev, clearance_radius_m=0.15)
    nav = p.embodiment_navigable_mask()
    assert not nav[11, 12]  # 10 cm from obstacle
    assert nav[12, 11]  # 20 cm from obstacle
    cost = Planner(bev)._traversal_cost(bev.navigable_mask())
    assert cost[11, 12] > cost[12, 11]


def test_world_exclusion_and_rectangular_astar(tmp_path):
    bev = make_map(tmp_path)
    x, y = bev.cell_to_world(6, 6)
    p = Planner(bev, world_exclusion_zones=[(x + 0.04, y, 0.075)])
    nav = p.active_navigable_mask()
    assert not nav[6, 6] and not nav[6, 7] and nav[7, 6]
    p = Planner(bev, simplify=False)
    nav = np.ones((22, 22), bool)
    p._active_cost = np.ones_like(nav, dtype=float)
    path = p._astar_numpy(nav, (3, 3), (5, 7))
    lengths = [
        np.hypot((b[0] - a[0]) * 0.2, (b[1] - a[1]) * 0.1)
        for a, b in zip(path, path[1:])
    ]
    assert np.isclose(sum(lengths), 2 * np.hypot(0.2, 0.1) + 2 * 0.1)

    # Two separate corridors: 18 grid steps with fewer expensive Y moves
    # must beat the 16-step route. An isotropic search chooses the latter.
    nav = np.zeros((12, 12), dtype=bool)
    nav[5, 0:6] = nav[5:10, 0] = nav[9, 0:10] = True
    nav[1:6, 5] = nav[1, 5:10] = nav[1:10, 9] = True
    p = Planner(bev, diagonal=False, simplify=False)
    p._active_cost = np.ones_like(nav, dtype=float)
    path = p._astar_numpy(nav, (5, 5), (9, 9))
    assert (5, 0) in path
    assert (1, 5) not in path
    distance = sum(
        np.hypot((b[0] - a[0]) * 0.2, (b[1] - a[1]) * 0.1)
        for a, b in zip(path, path[1:])
    )
    assert np.isclose(distance, 2.2)


def test_square_mask_policy_unchanged(tmp_path):
    bev = make_map(tmp_path, dy=0.1)
    p = Planner(bev, clearance_radius_m=0.15)
    assert not p._anisotropic
    np.testing.assert_array_equal(
        p.embodiment_navigable_mask(), erode_navigable_mask(bev.navigable_mask(), 2)
    )
