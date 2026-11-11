from types import SimpleNamespace as NS
from studies.proxy_physics_measurement import measure_proxy_physics


def test_reads_exact_existing_actor_with_explicit_units():
    calls = []
    actor = NS(count=1, get_transforms=lambda: [[100, 20, 0, 0, 0, 0, 1]])
    world = NS(physics_sim_view=NS(create_rigid_body_view=lambda path: calls.append(path) or actor),
               backend='numpy', backend_utils=NS(to_numpy=lambda value: value))
    result = measure_proxy_physics(world, '/World/Proxy/Nora', .01)
    assert calls == ['/World/Proxy/Nora']
    assert result['pose']['position_m'] == [1, .2, 0]
    assert result['pose']['quaternion_xyzw'] == [0, 0, 0, 1]
    actor.count = 0
    missing = measure_proxy_physics(world, '/World/Proxy/Nora', .01)
    assert missing['status'] == 'unavailable' and missing['count'] == 0
    assert missing['pose'] is None


def test_missing_world_view_never_uses_other_pose():
    receipt = measure_proxy_physics(NS(), '/World/Proxy/Nora', 1)
    assert receipt['status'] == 'unavailable'
    assert receipt['pose'] is None
    assert 'World.physics_sim_view' in receipt['error']
