# Explicit Isaac Sim 6.2 SDK launch

Use a provisioned Python 3.12 Isaac Sim 6.2 SDK environment. Do not source the
5.1 conda activation script or manufacture a VERSION file. The environment must
expose the genuine installed `isaacsim` distribution metadata and SDK USD modules.
Set asset paths, checkpoint paths, Vulkan ICD, provider endpoints and output path
outside the repository. Preserve the author's episode YAML and decision budget.

Model services can use separate interpreters through `TARSPHERE_SAM3_PYTHON`,
`TARSPHERE_GRASPGENX_PYTHON`, and `TARSPHERE_XNAVDP_PYTHON`. Set these to the pinned
model environment; leave the simulator's NumPy installation untouched.

`TARSPHERE_MODEL_SERVICE_GPU` is a physical GPU index; the service launcher
checks that it belongs to the caller's `CUDA_VISIBLE_DEVICES` allocation.
Resolve that index from the allocated device, rather than assuming index zero.
XNavDP forwards `TARSPHERE_MODEL_GPU` directly to its `CUDA_VISIBLE_DEVICES`.

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
`GRIPPER_CFG_DIR`; configure these in the service environment to point to the
provisioned resources and avoid implicit downloads or cloning.
