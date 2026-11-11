from types import SimpleNamespace
from tarsphere.infra.simulation.scene_query import SurfaceRayHit
from studies.proxy_scene_measurement import measure_proxy_rays

def Hit(path):
    return SurfaceRayHit(prim_path=path, position_xyz=(0,0,0), normal_xyz=(1,0,0))


def test_scene_query_retains_occluders_and_exact_proxy_at_both_locations():
    calls = []
    def all_hits(origin, direction, distance):
        calls.append(origin)
        return (Hit('/World/Other'), Hit('/World/Proxy'))
    receipt = measure_proxy_rays(SimpleNamespace(all_hits=all_hits), '/World/Proxy',
                                 [2, 0, 1], [0, 0, 1], .3, 1.8)
    assert receipt['status'] == 'available'
    rays = receipt['value']['rays']
    assert len(rays) == 12
    assert rays[0]['origin'][0] - rays[6]['origin'][0] == 2
    assert all(len(ray['hits']) == 2 and [hit['prim_path'] for hit in ray['exact_proxy_hits']] == ['/World/Proxy'] for ray in rays)
