# Explicit Isaac Sim 6.2 SDK launch

Use a provisioned Python 3.12 Isaac Sim 6.2 SDK environment. Do not source the
5.1 conda activation script or manufacture a VERSION file. The environment must
expose the genuine installed `isaacsim` distribution metadata and SDK USD modules.
Set asset paths, checkpoint paths, Vulkan ICD, provider endpoints and output path
outside the repository. Preserve the author's episode YAML and decision budget.

Model services can use separate interpreters through `TARSPHERE_SAM3_PYTHON`,
`TARSPHERE_GRASPGENX_PYTHON`, and `TARSPHERE_XNAVDP_PYTHON`. Set these to the pinned
model environment; leave the simulator's NumPy installation untouched.

`TARSPHERE_MODEL_SERVICE_GPU` is an NVML GPU index as seen inside the execution
container; the service launcher checks that it belongs to the caller's
`CUDA_VISIBLE_DEVICES` allocation. Resolve it by the allocated GPU UUID.
The host device number can differ from the container's index.
XNavDP forwards `TARSPHERE_MODEL_GPU` directly to its `CUDA_VISIBLE_DEVICES`.

Export `TARSPHERE_XNAVDP_ROOT` to the provisioned official X-NavDP checkout
(containing `eval/src/client_utils.py`) in the parent environment. The server
script's default path does not configure the worker client.

From the repository root, with `TARSPHERE_PYTHON` set to the SDK interpreter:

```bash
"$TARSPHERE_PYTHON" tarsphere/cli/check_environment.py --runtime isaac62 \
  --episodes episodes/case/navigate_and_report/task_update.yaml
bash tarsphere/infra/services/start.sh
"$TARSPHERE_PYTHON" tarsphere/cli/evaluate.py --launch \
  --episodes episodes/case/navigate_and_report/task_update.yaml \
  --model "$TARSPHERE_AGENT_MODEL" --api-base "$TARSPHERE_AGENT_API_BASE" \
  --episode-id home_navigate_and_report_task_update \
  --workers-per-gpu 1 --max-starting-workers 1 \
  --max-tokens 2048 --output-dir "$TARSPHERE_RUN_OUTPUT"
```

Run each command only after its predecessor succeeds. Use a new output directory
for each attempt. Service startup retains its health checks; evaluator startup
retains provider and worker readiness checks. Set a completion budget consistent
with the provider's context limit; 2048 is an example for an 8192 context with the
observed roughly 5281-token initial prompt, not a guarantee for later observations.

The explicit 6.2 preflight keeps source, episode assets, material composition,
robot camera, checkpoints and Vulkan checks. The 23 GiB default-runtime GPU memory
requirement remains mandatory. Passing preflight establishes installation conditions, not successful
navigation, model inference, task completion or compatibility of every SDK API.

Select the intended simulation GPU through CUDA visibility outside the application;
the one-worker settings above do not constrain the number of visible GPUs.
For external GraspGenX resources set `TARSPHERE_GRASPGENX_CHECKPOINT_ROOT` and
`TARSPHERE_GRASPGENX_GRIPPER_ASSETS`; startup forwards these explicit CLI paths.
Separately, upstream model construction may use `GRASPGENX_CHECKPOINT_DIR` and
`GRASPGENX_GRIPPER_CFG_DIR`; configure these in the service environment to point to the
provisioned resources and avoid implicit downloads or cloning.

For animated characters, stage the SDK `HumanMotionLibrary.usd` and export
`TARSPHERE_BEHAVIOR_MOTION_LIBRARY` to that file. Isaac 6.2 uses the SDK behavior
agent when the legacy People extension is absent. GoTo, Idle, Listen and
speech are implemented; other action requests fail explicitly. The composed
GoTo final-facing lifecycle has CPU checks but remains unverified in simulation. See
[character and manipulation limits](ISAAC_SIM_6_2.md#isaac-62-character-execution).

Provision Shapely 2.2 with constrained Delaunay support, `manifold3d==3.5.2`
and `trimesh>=4.0` in the simulator environment before launch. The runner probes
real triangulation and Boolean geometry before and after Kit startup to detect
vendored dependency replacement. For the table grasp component entrypoint and
its required prepared inputs, see [the diagnostic](ISAAC_SIM_6_2.md#table-grasp-diagnostic-and-geometry-dependencies).
No physical grasp success is established by environment preflight.

The grasp planner requires cuRobo 0.8.0. The audited bundled source is documented
in [TARSPHERE_SOURCE.md](../third_party/curobo/TARSPHERE_SOURCE.md). Its exact
`v0.8.0-no-tag` version sentinel is accepted only when imported from that bundled
checkout; it does not authorize an arbitrary unversioned installation. The
isolated worker's `initialize` operation constructs the hand planner and runs
its dependency warmup. Successful initialization establishes dependency
construction and warmup only; it does not plan or execute a physical grasp.

Before starting the interpreter, set the standard `TMPDIR` variable to a fresh
writable attempt directory outside the checkout. Worker request files retained
there after a failure can then be inspected and replayed using the same declared
inputs. Keep each attempt's temporary files with its outputs.
