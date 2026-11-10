# Isaac Sim 6.2 compatibility

This fork branch contains the compatibility changes tested in humanoid-office.
The original author baseline is `ea708be666d13962f7c74dbcd0360d34eb4d379a`;
the executed compatibility snapshot was `78e7d5d0323bac228c30844d9cf2a58a4672202e`.
The source changes below preserve that executed code. They establish the G1
body/camera check, not the complete original Isaac Sim 5.1 benchmark.

The check completed cold preparation, resident reset and 120 environment `step`
calls, with an upright endpoint assertion and three 640 by 480 RGB-D samples.
At all three samples, pelvis positions in USD and PhysX agreed and camera poses
and depth changed with body state. This does not establish continuous uprightness,
NPC interaction, navigation or manipulation.

[Detailed conditions and retained failures](https://github.com/LingzheZhao/humanoid-office/blob/58ced94/docs/results/interaction.md).

## Running the camera check

The tested condition is Python 3.12.15, Isaac Sim 6.2.0.0, Isaac Lab 3.0.0,
Torch 2.11 and NumPy 2.4.6 on an RTX 5060 Ti. It is a compatibility result,
separate from the author's original Isaac Sim 5.1 environment. Use the installed
Lab `isaaclab.python.headless.rendering.kit` and all installed SDK extension roots,
including `extsDeprecated`. This branch discovers these from installed packages.
Do not source the original 5.1 Conda activation into this environment.

Additional isolated Python dependencies used were `onnxruntime==1.19.2`,
`addict==2.4.0`, `lcm==1.5.0`, and the author's `rsl-rl-lib==2.2.1` distribution.
Install with `--no-deps` into a separate overlay after checking the existing
runtime, preserving its Torch/NumPy. Also install the genuine NVIDIA
`isaacsim-test==6.2.0.0` wheel into a separate target; set
`TARSPHERE_EXTRA_EXTENSION_ROOT` to that target's `isaacsim/exts` directory.
Its real docstring extension is required by the installed camera dependency chain.

Set `TARSPHERE_ASSETS_PATH` to the author's prepared asset root containing the
G1 USD, original `g1_15000.onnx` policy, and configured Dex3 donor assets. Assets
retain their separate licenses. Put the source and dependency overlay on
`PYTHONPATH`, together with the installed SDK's single `omni.usd.libs-*` folder.
The matching USD `bin`, SDK `kit`, and current Python library directory must be
available to the dynamic loader; do not add libraries from an older simulator.

Set `TARSPHERE_LOCAL_SCRATCH` to a new writable absolute directory. Set
`XDG_CACHE_HOME`, `OMNI_USER_CACHE`, and `TORCHINDUCTOR_CACHE_DIR` beneath it.
The runner supplies this directory as Kit's portable root; a root-owned installed
Kit cache caused actual renderer initialization failures. Supply a valid `USER`
and `LOGNAME` if the container UID lacks a passwd entry. Accept the NVIDIA EULA
through its normal environment variables and expose graphics driver capabilities.

Within an explicitly allocated GPU environment, bind `CUDA_VISIBLE_DEVICES` to
the allocated GPU UUID and pass its NVML physical index as `--gpu`; Kit uses that
physical index while PhysX uses logical device zero. Run with a finite timeout
and a fresh output directory:

```sh
timeout -k 10 900 python -c 'from isaacsim import SimulationApp; import runpy; runpy.run_module("tarsphere.cli.smoke_robot_camera", run_name="__main__")' --steps 120 --gpu "$ALLOCATED_PHYSICAL_GPU_INDEX" --output-dir /path/to/new-output
```

The implementation uses real classic APIs and public physics initialization, skips People
asset initialization only for an empty human cast, and explicitly applies the
author's `use_fabric=False` through the public SimulationManager API. The latter
restores USD body/camera transforms; the Kit update-to-USD flag alone was
insufficient. Immediate RGB/depth copies, timestamps and USD/PhysX pose checks
are diagnostics, not policy changes. Original G1 weights, control interval and
120-step criterion are unchanged. The verified result covers cold and resident
resets, standing control and three fresh RGB-D observations. Populated NPC casts,
full navigation/social tasks and manipulation remain untested by this check.

## Nora component diagnostic

The separate `studies/nora_component_diagnostic.yaml` selects the author's task-update
scene and public Nora fact. It holds the G1 with its original policy and commands
Nora along two authored destinations through the existing occupancy planner.
The default NPC backend translates the character root with an idle skeleton;
this is not animated walking. The diagnostic ends when both routes complete or
the configured time limit is reached. It records the author's step counter and
the simulator clock separately.

With the complete scene, character, material and occupancy assets prepared and
the support-model endpoint configured, run from the repository root:

```sh
python studies/nora_component_diagnostic.py \
  --config studies/nora_component_diagnostic.yaml --output /absolute/new-diagnostic
```

The diagnostic uses real model fact selection and actual human-body command
admission. It does not install the episode runner's robot-heard inbox, assign
benchmark milestones, or estimate agent-policy performance. Speech admission and
proximity alone are not recorded as confirmed robot receipt. Captured arrays,
poses, available contact-monitor readings and partial outcomes remain outside
the source tree. This diagnostic still requires physical validation.
