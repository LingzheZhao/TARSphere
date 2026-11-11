# Isaac Sim 6.2 compatibility

This fork branch contains the compatibility changes tested in humanoid-office.
The original author baseline is `ea708be666d13962f7c74dbcd0360d34eb4d379a`;
the executed compatibility snapshot was `78e7d5d0323bac228c30844d9cf2a58a4672202e`.
That snapshot establishes the G1
body/camera check, not the complete original Isaac Sim 5.1 benchmark. Subsequent
interfaces described below have separate validation limits.

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

The first full-scene attempts exposed missing textures and relative MDL imports,
then the removed `add_update_semantics` API during character creation. Material
preparation now follows relative module dependencies and their textures; missing
dependencies must be supplied by the original collection. Character identities
use the USD `LabelsAPI` read by Isaac Sim 6.2. A warmup camera frame from a scene
with shader compilation errors does not establish a valid visual observation.

A subsequent attempt captured Nora in the robot camera and registered a real
model-selected text response, then stopped before movement because the released
occupancy map has different X and Y cell spacing. The planner now uses those
actual spacings for distance, clearance and A* costs, preserving the existing
square-grid behavior. Default kinematic routes also avoid the live robot using
the author's existing robot–human motion clearance. CPU queries on the released
map reach both exact destinations with that exclusion; actual execution remains
to be verified.

Texture inputs now use resolved absolute paths. Computing relative paths from
canonical directories failed when USD opened the same layers through symlink
aliases at another directory depth. A real USD composition regression covers
that failure. The focused CPU suite passes 89 tests.

The diagnostic separately records the author step counter, World clock and
PhysX post-step callbacks. In the observed 6.2 condition their readings differ;
the World clock accumulates callback step sizes. Neither counter is treated as
independent evidence of the robot scene's integration time. The diagnostic's
time bounds and control sequence remain unchanged.

## Isaac 6.2 character execution

The `animation` human-motion backend selects `BehaviorHumanExecutor` when the
installed SDK provides `omni.anim.behavior.core` without `omni.anim.people`.
SDKs with the legacy People extension retain `AnimPeopleBodyExecutor`. The
`kinematic` backend described above remains a separate root-translation diagnostic.
Teleoperation selects `animation`; it does not select the kinematic substitute.

Stage the SDK's `HumanMotionLibrary.usd` and set
`TARSPHERE_BEHAVIOR_MOTION_LIBRARY` to its file path. The executor uses the
original `F_Medical_01` character, explicitly loads its USD payload, and binds
its skeleton through the SDK control-rig mapping and behavior agent. It drives
SDK tasks rather than writing walking root transforms or synthesizing joints.
GoTo, Idle, Listen and speech are implemented; other actions fail explicitly.
GoTo final facing composes move-to and idle-facing SDK tasks into one completion
lifecycle. That composition has CPU checks but has not been verified in simulation. Sitting and the full author animation action set
are not validated on this backend.

The 19-second out-and-return protocol followed by a standalone 90-degree
idle-facing request completed on revision `20d02e6`. This did not exercise the
later composed GoTo final-facing lifecycle. Numeric joint transforms were unavailable in
that reading, so task completion does not establish measured skeletal fidelity
or full animation coverage. Task feedback, rendered observations and skeleton
measurements have distinct evidential limits.

Compatibility paths use the public Kit application loop and PhysX backend
`resolve_indices` interface. These interface fixes do not establish physical
manipulation success.

## Table grasp diagnostic and geometry dependencies

`examples/table_grasp_diagnostic.py` runs the author's public table grasp skill
without an MLLM. Supply a CPU-prepared `GroundedScene` JSON and its reviewed
occupancy file under the declared assets root:

```sh
python examples/table_grasp_diagnostic.py \
  --assets-root /absolute/prepared-assets \
  --grounded-scene /absolute/prepared-grounded-scene.json \
  --request-id table-attempt-1 --max-steps 12 \
  --output-root /absolute/new-table-output
```

Run in a provisioned SDK/model environment with a finite external wall-clock
budget. The diagnostic refuses existing output roots, records declared inputs,
retains failures and reports holding, retention and transport samples only when
present. Exit status alone is not physical grasp evidence. This component check
does not replace Fetch-and-Deliver execution.

Support geometry requires Shapely with constrained Delaunay triangulation
(the source minimum is 2.1.2; the provisioned condition uses 2.2),
`manifold3d==3.5.2` and `trimesh>=4.0`. Preload these before Kit can prepend its
older vendored Shapely. The standard runner exercises triangulation and the
Mesh64 Boolean/export path before Kit, then repeats the probe after startup and
checks that dependency identities did not change. The table diagnostic performs
the preload before constructing its scene server. Import success alone is
insufficient evidence that support geometry works.

Declared pickables loaded from `scene_static` are normalized as held kinematic
rigid bodies before the first physics advance, including scene baking. The
ordinary grasp lifecycle subsequently controls their physical release. This
prevents initialization from dropping objects before the test begins; it does
not establish successful contact, retention, lift or transport. Manipulation
remains blocked by infrastructure failures; no physical grasp success has been
validated in this Isaac 6.2 condition.

## Published benchmark definitions

The repository's twelve `episodes/case` files are schema examples, not the
paper's complete suite. The [author benchmark explorer](https://xuyicheng-zju.github.io/TARSphere/benchmark.html)
publicly exposes 360 authored definitions across 30 scenes. Retrieve them into
a fresh directory outside the checkout:

```sh
python -m studies.fetch_published_episodes /absolute/external/new-benchmark-root
scripts/evaluate.sh --plan --episodes /absolute/external/new-benchmark-root/index.json
```

The downloader captures the actual page, index and detail script, preserves each
raw author JSON export and its source reference, writes each embedded definition
as YAML, and validates it with the installed schema loader. It refuses existing
output roots, uses at most eight HTTP workers and bounded request timeouts, and
writes its acquisition time and URLs beside the outputs. A failed acquisition
leaves its partial root visible; use a distinct fresh root for another attempt.
The website is mutable; captured inputs establish what was retrieved, rather
than a permanently pinned website revision. Downloads never belong in Git.

The 2026-11-11 acquisition contained 360 unique episode IDs, 180 per task family
and 60 per condition. All passed the original pinned author schema loader.
The provisioned `MVUCSQAKTKJ5EAABAAAAABY8_usd` scene corresponds to the twelve
`corner_lounge_home` episodes; the README scene
`MV7J6NIKTKJZ2AABAAAAADY8_usd` corresponds to `sectional_sofa_home`.
Repository cases have different identities and definitions: earlier example
runs cannot be relabeled as published benchmark runs. Schema validation does
not establish scene, character, metadata or model resource availability, or
physical execution success. Run the episode-specific asset preflight separately.

## Published navigation-and-report result

Revision `806e3e7` completed the published
`corner_lounge_home_navigate_and_report_base` task on Isaac Sim 6.2 with
`gpt-6.1-sol` through Codex at medium effort and Qwen3-4B support: **5/5,
success**, 26 decisions, 26 actions, no fall. The robot inspected the bed,
retained its submitted finding, returned and delivered the correct report to
Elena. The matched `gpt-6.1-terra` condition ended in a physical fall at **1/5**.
Both used an RTX 3090, the same task/assets/controller, 100-decision cap and one
physical worker. This is one paired episode, not a general model ranking.

The successful condition includes an explicit public inspection-output reminder
and durable retention of the agent's own admitted inspection statements. Those
statements remain historical context, not new scoring events or evaluator truth.
The NPC classifier also skips empty topic sets and validates other selections
within its existing bounded retries. Sixty-two focused CPU checks cover these
contracts. Earlier unmodified-prompt failures remain separate evidence.

All final videos decode completely, including the failed trajectory. The
successful run still recorded scene contacts; moving-person navigation,
fetch-and-deliver and the complete benchmark are not validated by this result.
This port also differs from the author's SDK and model-service conditions, so
it does not reproduce the paper's benchmark scores. [Detailed evidence](https://github.com/LingzheZhao/humanoid-office/blob/codex/bootstrap/docs/results/interaction.md#paired-published-task-after-inspection-retention).
