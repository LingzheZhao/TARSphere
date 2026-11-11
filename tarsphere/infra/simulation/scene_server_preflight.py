"""Preload the default navigation dependencies before Isaac starts."""

from __future__ import annotations


def preload_open_frontier_clustering_dependencies() -> dict[str, str]:
    """Load and exercise HDBSCAN before Kit adds its vendored SciPy paths.

    Isaac Sim ships a partial SciPy tree for its own extensions. Once Kit is
    live that tree can be resolved together with the environment's sklearn
    and hdbscan, producing an ABI-incoherent mixed import. OpenFrontier's
    clustering is CPU-only, so load its declared dependency set before the
    SimulationApp exists and execute one tiny fit to resolve lazy imports.
    """

    try:
        import hdbscan
        import numpy as np
        import scipy
        import sklearn

        probe = np.column_stack(
            [
                np.linspace(0.0, 1.0, 32, dtype=np.float64),
                np.zeros((32, 4), dtype=np.float64),
            ]
        )
        labels = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
        ).fit_predict(probe)
        if np.asarray(labels).shape != (32,):
            raise RuntimeError("HDBSCAN smoke fit returned an invalid shape.")
    except Exception as exc:
        raise RuntimeError(
            "OpenFrontier requires one coherent numpy/scipy/sklearn/hdbscan "
            "environment before Isaac starts."
        ) from exc
    return {
        "hdbscan": str(getattr(hdbscan, "__file__", "")),
        "scipy": str(getattr(scipy, "__file__", "")),
        "sklearn": str(getattr(sklearn, "__file__", "")),
    }


def preload_support_surface_geometry() -> dict[str, str]:
    """Resolve project Shapely before Kit prepends its older vendored copy.

    Exercise the GEOS operation used by support-surface triangulation, not just
    the import. This CPU-only dependency must never be replaced after Kit starts.
    """
    try:
        import shapely
        from importlib.metadata import version
        from packaging.version import Version
        from shapely import constrained_delaunay_triangles
        import numpy as np
        import manifold3d as mf
        import trimesh
        from shapely.geometry import Polygon
        import shapely.ops  # Resolve the support-surface module's imports now.
        import shapely.geometry.polygon

        triangles = constrained_delaunay_triangles(Polygon([(0, 0), (1, 0), (0, 1)]))
        if len(triangles.geoms) != 1:
            raise RuntimeError("Shapely constrained triangulation returned an invalid probe")
        if Version(shapely.__version__) < Version("2.1.2"):
            raise RuntimeError("Shapely>=2.1.2 is required")
        if version("manifold3d") != "3.5.2" or Version(trimesh.__version__) < Version("4.0"):
            raise RuntimeError("manifold3d==3.5.2 and trimesh>=4.0 are required")
        # Same Mesh64/Boolean/export path used by the real support envelope.
        vertices = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1],
                             [0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([(0, 1, 2), (5, 4, 3), (0, 3, 4), (0, 4, 1),
                          (1, 4, 5), (1, 5, 2), (2, 5, 3), (2, 3, 0)], dtype=np.uint64)
        prisms = [mf.Manifold(mf.Mesh64(vertices + shift, faces))
                  for shift in (0.0, np.array([0.25, 0.0, 0.0]))]
        if any(prism.status() != mf.Error.NoError for prism in prisms):
            raise RuntimeError("Manifold support-prism probe is invalid")
        solid = mf.Manifold.batch_boolean(prisms, mf.OpType.Add).as_original().simplify(1e-7)
        mesh = solid.to_mesh64()
        surface_mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3],
                                 faces=np.asarray(mesh.tri_verts), process=False)
        if solid.status() != mf.Error.NoError or not surface_mesh.is_watertight or surface_mesh.volume <= 0:
            raise RuntimeError("Support Boolean/Trimesh probe returned invalid geometry")
    except Exception as exc:
        raise RuntimeError(
            "Support-surface geometry requires the project Shapely>=2.1.2 "
            "with constrained Delaunay support, manifold3d==3.5.2 and trimesh>=4.0 "
            "before Isaac starts"
        ) from exc
    return {"version": str(shapely.__version__), "path": str(shapely.__file__),
            "manifold3d_version": version("manifold3d"), "manifold3d_path": str(mf.__file__),
            "trimesh_version": str(trimesh.__version__), "trimesh_path": str(trimesh.__file__),
            "numpy_version": str(np.__version__), "numpy_path": str(np.__file__)}
