"""Bounded public-image SAM service diagnostic; overlays require human review.

Reports the service's current filtering without adding confidence thresholds.
Negative controls describe an expected occluded object, not proof of absence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from tarsphere.harness.perception.remote_backends import SAM3ServiceClient


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def run(input_dir: Path, output_dir: Path, config: Path, client) -> dict:
    cases = yaml.safe_load(config.read_text())["cases"]
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {"config": yaml.safe_load(config.read_text()), "cases": [],
               "interpretation": "Structural positives require visual mask review; no confidence threshold added."}
    def save():
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    save()
    try:
        summary["readiness"] = client.request({"cmd": "sam3_require_ready"})
        for case_index, case in enumerate(cases):
            rgb = load_rgb(input_dir / case["image"])
            Image.fromarray(rgb).save(output_dir / f"image-{case_index}.png")
            for expected, prompts in (("positive", case["positive"]), ("negative", case["negative"])):
                for prompt in prompts:
                    index = len(summary["cases"])
                    result = client.request({"cmd": "sam3_process", "rgb": rgb,
                                             "text_prompts": [prompt]})
                    record = {"image": case["image"], "prompt": prompt,
                              "expected": expected, "rgb_shape": list(rgb.shape),
                              "detections": [], "tool_trace": result.tool_trace,
                              "metadata": result.metadata}
                    for di, detection in enumerate(result.detections):
                        mask = None if detection.mask is None else np.asarray(detection.mask)
                        valid = mask is not None and mask.shape == rgb.shape[:2]
                        item = {"label": detection.label, "score": float(detection.score),
                                "box_xyxy": list(detection.box_xyxy), "source": detection.source,
                                "mask_shape": None if mask is None else list(mask.shape),
                                "mask_dtype": None if mask is None else str(mask.dtype),
                                "mask_nonzero": None if mask is None else int(np.count_nonzero(mask)),
                                "valid_nonempty_mask": bool(valid and np.any(mask))}
                        if mask is not None:
                            np.save(output_dir / f"case-{index}-mask-{di}.npy", mask, allow_pickle=False)
                            if valid:
                                overlay = rgb.copy()
                                selected = mask.astype(bool)
                                overlay[selected] = (0.5 * rgb[selected] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
                                Image.fromarray(overlay).save(output_dir / f"case-{index}-overlay-{di}.png")
                        record["detections"].append(item)
                    record["detection_count"] = len(record["detections"])
                    record["structural_positive"] = any(d["valid_nonempty_mask"] for d in record["detections"])
                    summary["cases"].append(record)
                    save()
        positives = [r for r in summary["cases"] if r["expected"] == "positive"]
        summary["all_positive_controls_structurally_pass"] = bool(positives) and all(r["structural_positive"] for r in positives)
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_suffix(".yaml"))
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.config,
        SAM3ServiceClient(timeout_seconds=120))


if __name__ == "__main__":
    main()
