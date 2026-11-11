from types import SimpleNamespace
from dataclasses import dataclass
from studies.proxy_scene_measurement import measure_proxy_rays


@dataclass
class Hit:
    prim_path: str


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
    assert all(len(ray['hits']) == 2 and ray['exact_proxy_hits'] == [{'prim_path': '/World/Proxy'}] for ray in rays)
