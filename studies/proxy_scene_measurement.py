"""Independent read-only PhysX scene-query receipts for the exact proxy shape."""


def measure_proxy_rays(query, path, current_center, initial_center, radius, height):
    from tarsphere.infra.control.human_native_motion_trace import measured
    def read():
        if not path or radius <= 0 or height <= 0:
            raise ValueError('Exact proxy path and positive capsule dimensions required')
        live_query = query() if callable(query) else query
        rays = []
        # Retain current and initial locations independently, even when coincident.
        for label, center in [('current', current_center), ('initial', initial_center)]:
            for offset in (-height / 4, 0, height / 4):
                for axis in (0, 1):
                    origin = list(map(float, center))
                    origin[2] += offset
                    origin[axis] -= 2 * radius
                    direction = [0., 0., 0.]
                    direction[axis] = 1.
                    hits = live_query.all_hits(tuple(origin), tuple(direction), 4 * radius)
                    rays.append(dict(location=label, origin=origin, direction=direction,
                        distance=4 * radius, hits=[hit.model_dump(mode='json') for hit in hits],
                        exact_proxy_hits=[hit.model_dump(mode='json') for hit in hits if hit.prim_path == path]))
        return dict(path=path, expected_radius=radius, expected_height=height,
                    current_center=list(current_center), initial_center=list(initial_center), rays=rays)
    return measured(read)
