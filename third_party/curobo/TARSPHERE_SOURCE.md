# cuRobo source

- Upstream: [NVlabs/curobo](https://github.com/NVlabs/curobo).
- Release: `v0.8.0`, commit `4ea77366ca48ee453e7df139e39fa6532af49f3b`.
- Licenses: [Apache-2.0 source license](LICENSE) and [asset-specific terms](LICENSE_ASSETS).

A bytewise comparison against that official tag found 776 identical files.
The differing source files are `curobo/_src/geom/data/data_mesh.py` and
`curobo/_src/perception/mapper/mesh_extractor.py`: they adapt
`wp.context.Device` to `wp.Device` and `wp.torch.device_from_torch` to
`wp.device_from_torch` for the installed Warp API. The latter file also differs
in trailing whitespace. README differs; upstream documentation, contributor,
GitHub and agent instruction files are omitted. No other upstream source or
resource file differs or is missing in that comparison.

The fixed TARSphere revision identifies this bundled source plus its Warp
adaptations. CUDA extensions are built on the target machine. Without Git
metadata or an installed `nvidia_curobo` distribution, the unchanged official
version helper reports `v0.8.0-no-tag`; TARSphere accepts that sentinel only
when the loaded package is this checkout's bundled package. External installs
must report the supported release through ordinary version metadata.
