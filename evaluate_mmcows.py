#!/usr/bin/env python
"""Stage-7 evaluator for VFA on multi-sequence MmCows.

This file intentionally reuses ``evaluate.py::resume`` for Stage-6 checkpoint
loading.  Do not duplicate that function here; keeping one resume path avoids
silent model-construction drift between the original VFA evaluator and this
BEVine3D-compatible evaluator.

Data path:
  * Stage-4 ``vfa.data.multiseq_dataset.build_multiseq``, restricted by
    default to ``mmcows_test_only``, with batch size 1, no shuffling, and
    ``vfa.utils.collate``.  If the requested split does not contain that
    sequence, the evaluator switches to the manifest split that does.

Detection/GT convention:
  * VFA/MmCows adapter locations and decoded predictions are already in the
    shifted VFA grid frame: x = world_x - (-879), y = world_y - (-646), in cm.
    Therefore the BEVine3D 10-cm grid coordinates are simply x/10 and y/10.
  * GT rows are rounded 3D centroids.  This mirrors WorldTrack
    ``datasets/pedestrian_dataset.py::PedestrianDataset.prepare_gt`` with
    ``center_from_3d=True``, which calls
    ``MmCows.get_worldgrid_from_center()`` and writes ``int(round(grid_x))``,
    ``int(round(grid_y))``.
  * Prediction rows are NOT rounded: BEVine3D's prediction path stores
    floating-point grid positions.

Metric path:
  * WorldTrack ``evaluation.mod.modMetricsCalculator(det, gt, td_cells=10.0)``.
    On the 10-cm MmCows grid, 10 cells is exactly the 100-cm gate.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# vfa/visualization/bbox.py and RGK.py are VFA-original and use np.int.
if not hasattr(np, "int"):
    np.int = int

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from vfa.config import mmcows_opts
from vfa.data.encoder import ObjectEncoder
from vfa.data.mmcows import WORLD_X_MIN_CM, WORLD_Y_MIN_CM
from vfa.data.multiseq_dataset import MultiSeqFrameDataset, build_multiseq
from vfa.utils import collate, to_numpy
from vfa.visualization.figure import visualize_bboxes

EVAL_GRID_CELL_CM = 10.0
EVAL_GATE_CM = 100.0
EVAL_TD_CELLS = EVAL_GATE_CM / EVAL_GRID_CELL_CM
FRAME_SEQ_OFFSET = 1_000_000
EXPECTED_WORLD_SIZE = (1200, 1925)       # (y extent, x extent), cm
EXPECTED_CUBE_LWH = (25, 25, 32)         # cm
EXPECTED_REDUCED_GRID = [48, 77]         # (y cells, x cells)
DEFAULT_EVAL_SEQUENCE = "mmcows_test_only"


# -----------------------------------------------------------------------------
# CLI and environment
# -----------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VFA on MmCows with BEVine3D's point-MOD evaluator."
    )
    parser.add_argument("--data", default="MmCows",
                        help="Only MmCows is supported by this evaluator.")
    parser.add_argument("--root", default=mmcows_opts.root,
                        help="MmCows multi-sequence DATA root.")
    parser.add_argument("--manifest", default=mmcows_opts.manifest_name,
                        help="Split manifest inside --root.")
    parser.add_argument("--split", choices=("val", "test"), default="test",
                        help="Preferred manifest split. If --sequence is in a "
                             "different split, the split containing that "
                             "sequence is used and reported.")
    parser.add_argument("--sequence", default=DEFAULT_EVAL_SEQUENCE,
                        help="Single sequence to evaluate. Use --sequence '' "
                             "to evaluate every sequence in --split. Default: "
                             "mmcows_test_only.")
    parser.add_argument("--checkpoint", required=True,
                        help="Stage-6 .pth checkpoint path.")
    parser.add_argument("--cls_thresh", type=float, default=0.70,
                        help="Decode confidence threshold for a single run.")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep cls_thresh and select the MODA maximum.")
    parser.add_argument("--sweep_min", type=float, default=0.10)
    parser.add_argument("--sweep_max", type=float, default=0.90)
    parser.add_argument("--sweep_step", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=None,
                        help="Decoder top-k. Default: checkpoint args.topk, "
                             "then 100.")
    parser.add_argument("--resize_size", type=int, nargs=2, default=None,
                        metavar=("H", "W"),
                        help="Evaluation image resize. Default: checkpoint "
                             "args.resize_size, then the MmCows config.")
    parser.add_argument("--worldtrack_root", default=None,
                        help="WorldTrack repository root. Default: "
                             "WORLDTRACK_ROOT or /home/mae/BEVine3D/WorldTrack.")
    parser.add_argument("--output_dir", default="experiments_mmcows/eval",
                        help="Evaluation output directory.")
    parser.add_argument("--run_tag", default=None,
                        help="Optional name used to namespace sweep outputs; "
                             "defaults to the checkpoint filename stem.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"),
                        default="auto",
                        help="Use --device cpu when the training GPU is full.")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Always write 5-frame diagnostic overlays and "
                             "confidence histograms.")
    parser.add_argument("--diagnose_frames", type=int, default=5,
                        help="Number of frames to decode for diagnostics.")
    parser.add_argument("--diagnostic_moda", type=float, default=1.0,
                        help="Auto-run diagnostics when overall MODA <= this.")
    parser.add_argument("--strict_gt_txt", action="store_true",
                        help="Fail if sequence gt.txt differs from the "
                             "adapter's prepare_gt(center_from_3d=True) rows. "
                             "Default: report the mismatch as a warning.")
    parser.add_argument("--skip_unit_check", action="store_true",
                        help="Do not verify GT/writer coordinates before "
                             "inference. Not recommended.")
    args = parser.parse_args(argv)

    if args.data != "MmCows":
        raise ValueError(f"evaluate_mmcows.py supports only MmCows, got {args.data!r}")
    if not 0.0 <= args.cls_thresh <= 1.0:
        raise ValueError(f"--cls_thresh must be in [0, 1], got {args.cls_thresh}")
    if args.sweep_step <= 0:
        raise ValueError("--sweep_step must be positive")
    if not (0.0 <= args.sweep_min <= args.sweep_max <= 1.0):
        raise ValueError("Require 0 <= sweep_min <= sweep_max <= 1")
    if args.diagnose_frames <= 0:
        raise ValueError("--diagnose_frames must be positive")
    return args


def resolve_worldtrack_root(path: str | None) -> str:
    root = path or os.environ.get("WORLDTRACK_ROOT") or "/home/mae/BEVine3D/WorldTrack"
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"WorldTrack repository root not found: {root!r}. Pass "
            "--worldtrack_root or set WORLDTRACK_ROOT."
        )
    os.environ["WORLDTRACK_ROOT"] = root
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def import_mod_metrics(worldtrack_root: str):
    """Import the SAME point-MOD evaluator used by BEVine3D."""
    if worldtrack_root not in sys.path:
        sys.path.insert(0, worldtrack_root)
    try:
        from evaluation.mod import modMetricsCalculator
    except ImportError as exc:
        raise ImportError(
            f"Could not import evaluation.mod from {worldtrack_root!r}. "
            "Expected WorldTrack/evaluation/mod.py."
        ) from exc
    return modMetricsCalculator


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(name)


def safe_tag(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text)


def checkpoint_metadata(path: Path) -> Tuple[argparse.Namespace | None, Dict]:
    """Read only checkpoint metadata for top-k / resize defaults."""
    try:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # older torch has no weights_only
            ckpt = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Could not read checkpoint metadata from {path}: {exc}") from exc
    if not isinstance(ckpt, dict):
        raise ValueError(f"Stage-6 checkpoint must be a dict, got {type(ckpt).__name__}")
    return ckpt.get("args"), ckpt

def load_resume_function(evaluate_path: Path):
    """Load ``evaluate.py::resume`` source without executing evaluate.py.

    The original evaluation script imports optional CUDA AP/AOS extensions at
    module scope.  Those extensions are unrelated to checkpoint construction
    and may not be compiled on the evaluation machine.  Extracting the
    top-level ``resume`` function reuses the exact maintained source while
    avoiding those unrelated imports.
    """
    if not evaluate_path.is_file():
        raise FileNotFoundError(
            f"evaluate_mmcows.py must sit next to evaluate.py; missing "
            f"{evaluate_path}"
        )
    source = evaluate_path.read_text()
    tree = ast.parse(source, filename=str(evaluate_path))
    resume_node = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "resume":
            resume_node = node
            break
    if resume_node is None:
        raise ImportError(f"No top-level resume() function found in {evaluate_path}")
    resume_source = ast.get_source_segment(source, resume_node)
    if resume_source is None:
        raise ImportError(f"Could not extract resume() source from {evaluate_path}")

    # resume() needs torch and VFANet from evaluate.py's module namespace.
    from vfa.model.vfanet import VFANet
    namespace = {"torch": torch, "VFANet": VFANet}
    # Preserve evaluate.py's original line numbers so tracebacks point at the
    # actual source line rather than the extracted snippet's first line.
    padded_source = "\n" * (resume_node.lineno - 1) + resume_source
    exec(compile(padded_source, str(evaluate_path), "exec"), namespace)
    return namespace["resume"]

@contextmanager
def checkpoint_unpickle_context():
    """Temporarily make torch.load use the legacy full unpickler.

    Stage-6 checkpoints are trusted local training artifacts.  They contain
    both argparse.Namespace and NumPy-serialized values, so PyTorch >=2.6's
    weights_only=True default rejects them.  Adding individual safe globals is
    brittle because NumPy reconstruction uses several internals; instead, force
    weights_only=False only for torch.load calls made by the inherited
    evaluate.py::resume() inside this context.
    """
    original_load = torch.load

    def trusted_stage6_load(*args, **kwargs):
        kwargs["weights_only"] = False
        try:
            return original_load(*args, **kwargs)
        except TypeError as exc:
            # PyTorch <2.0 does not expose the weights_only keyword.
            if "weights_only" not in str(exc):
                raise
            kwargs.pop("weights_only", None)
            return original_load(*args, **kwargs)

    torch.load = trusted_stage6_load
    try:
        yield
    finally:
        torch.load = original_load


def resume_stage6_model(checkpoint: Path, device: torch.device):
    """Load the model through evaluate.py::resume, unchanged."""
    evaluate_path = Path(__file__).resolve().with_name("evaluate.py")
    resume = load_resume_function(evaluate_path)
    try:
        # PyTorch >=2.6 defaults torch.load(..., weights_only=True).  Stage-6
        # checkpoints are trusted local artifacts containing argparse and NumPy
        # objects; use the legacy full unpickler only inside inherited resume().
        with checkpoint_unpickle_context():
            model = resume(str(checkpoint), device)
    except RuntimeError as exc:
        if "CUDA" in str(exc) or "cuda" in str(exc):
            raise RuntimeError(
                f"Could not load {checkpoint} on {device}. If this checkpoint "
                "was saved from CUDA and the GPU is occupied/unavailable, "
                "rerun on the GPU or patch evaluate.py::resume to call "
                "torch.load(..., map_location=device)."
            ) from exc
        raise
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Dataset and coordinate helpers
# -----------------------------------------------------------------------------
def locate_sequence_split(root: str, manifest_name: str, sequence: str,
                          preferred_split: str) -> str:
    """Find the manifest split containing ``sequence``.

    The requested split remains the preferred split if the sequence appears in
    more than one split.  Otherwise the evaluator uses the split that actually
    contains the requested sequence and says so loudly.  This prevents a
    `--split val --sequence mmcows_test_only` invocation from silently
    evaluating the wrong sequence.
    """
    manifest_path = Path(root) / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    with manifest_path.open("r") as f:
        manifest = json.load(f)

    containing = []
    for split in ("train", "val", "test"):
        names = [str(name) for name in manifest.get(split, [])]
        if sequence in names:
            containing.append(split)
    if not containing:
        available = {split: [str(name) for name in manifest.get(split, [])]
                     for split in ("train", "val", "test")}
        raise ValueError(
            f"Sequence {sequence!r} is not present in {manifest_path}. "
            f"Available sequences: {available}"
        )
    if preferred_split in containing:
        return preferred_split
    # Deterministic fallback when a sequence is duplicated unexpectedly.
    for split in ("test", "val", "train"):
        if split in containing:
            return split
    raise AssertionError("unreachable split lookup state")


def select_single_sequence(dataset: MultiSeqFrameDataset,
                           sequence: str) -> MultiSeqFrameDataset:
    """Return a one-sequence MultiSeqFrameDataset with reset boundaries."""
    matches = [i for i, name in enumerate(dataset.sequence_names)
               if name == sequence]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one sequence named {sequence!r} in split "
            f"{dataset.split!r}, found {len(matches)}. Available: "
            f"{dataset.sequence_names}"
        )
    idx = matches[0]
    selected = MultiSeqFrameDataset(
        [dataset.datasets[idx]],
        sequence_names=[dataset.sequence_names[idx]],
    )
    selected.split = dataset.split
    selected.manifest_path = getattr(dataset, "manifest_path", None)
    selected.requested_split = getattr(dataset, "requested_split", dataset.split)
    print(f"[SEQUENCE] Evaluating only {sequence!r} from split "
          f"{selected.split!r}: {len(selected)} frames.")
    return selected


def build_eval_dataset(args: argparse.Namespace,
                       resize_size: Tuple[int, int]) -> MultiSeqFrameDataset:
    transform = transforms.Compose([
        transforms.Resize(tuple(int(v) for v in resize_size)),
        transforms.ToTensor(),
    ])
    sequence = str(args.sequence).strip() if args.sequence is not None else ""
    requested_split = args.split
    eval_split = requested_split
    if sequence:
        eval_split = locate_sequence_split(
            args.root, args.manifest, sequence, requested_split)
        if eval_split != requested_split:
            print(f"[SEQUENCE] Requested split {requested_split!r}, but "
                  f"{sequence!r} is in manifest split {eval_split!r}; "
                  f"evaluating split {eval_split!r} instead.")

    dataset = build_multiseq(
        args.root,
        manifest_name=args.manifest,
        split=eval_split,
        transform=transform,
        drop_stationary=False,       # val/test are never filtered
        annotation_mode="3d",
        worldtrack_root=args.worldtrack_root,
        reload_heatmaps=False,
    )
    if not isinstance(dataset, MultiSeqFrameDataset):
        raise TypeError(f"Expected MultiSeqFrameDataset, got {type(dataset).__name__}")
    dataset.requested_split = requested_split
    if sequence:
        dataset = select_single_sequence(dataset, sequence)
    return dataset


def first_frame_dataset(dataset: MultiSeqFrameDataset):
    return dataset.datasets[0]


def verify_mmcows_geometry(dataset: MultiSeqFrameDataset) -> None:
    """Fail loudly if the Stage-1 geometry preamble is no longer true."""
    first = first_frame_dataset(dataset)
    errors = []
    if getattr(first.base, "__name__", None) != "MmCows":
        errors.append(f"first base dataset is {first.base.__name__!r}, not 'MmCows'")
    if tuple(first.world_size) != EXPECTED_WORLD_SIZE:
        errors.append(f"world_size={first.world_size}, expected {EXPECTED_WORLD_SIZE}")
    if tuple(int(v) for v in first.cube_LWH) != EXPECTED_CUBE_LWH:
        errors.append(f"cube_LWH={first.cube_LWH}, expected {EXPECTED_CUBE_LWH}")
    if list(first.reduced_grid_size) != EXPECTED_REDUCED_GRID:
        errors.append(
            f"reduced_grid_size={first.reduced_grid_size}, expected "
            f"{EXPECTED_REDUCED_GRID}"
        )
    grid_shape = tuple(first.grid.shape[:2])
    if grid_shape != tuple(EXPECTED_REDUCED_GRID):
        errors.append(f"make_grid shape={grid_shape}, expected {tuple(EXPECTED_REDUCED_GRID)}")

    # The writer assumes decoded x/y are relative to (-879, -646), not raw
    # world coordinates.  Check that invariant directly from the preamble
    # constants used by the adapter and the projector.
    if (WORLD_X_MIN_CM, WORLD_Y_MIN_CM) != (-879.0, -646.0):
        errors.append(
            f"MmCows origin constants changed to {(WORLD_X_MIN_CM, WORLD_Y_MIN_CM)}"
        )
    if errors:
        raise RuntimeError(
            "MmCows/VFA geometry check failed; do not evaluate until the "
            "writer is re-derived:\n  - " + "\n  - ".join(errors)
        )
    print("[GEOMETRY] OK: decoded/GT x,y are cm relative to (-879, -646); "
          "world_size=(1200,1925), grid=(48,77), cube=(25,25,32).")


def sequence_for_global_index(dataset: MultiSeqFrameDataset,
                              global_idx: int) -> Tuple[int, str, int]:
    for seq_idx, ((start, end), name) in enumerate(
            zip(dataset.sequence_boundaries, dataset.sequence_names)):
        if start <= global_idx < end:
            return seq_idx, name, global_idx - start
    raise IndexError(f"global index {global_idx} outside dataset of {len(dataset)} frames")


def object_xy_cm(objects: Iterable) -> np.ndarray:
    xy = []
    for obj in objects:
        loc = np.asarray(to_numpy(obj.location), dtype=np.float64).reshape(-1)
        if loc.size < 2:
            continue
        xy.append(loc[:2])
    if not xy:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack(xy, axis=0)


def prediction_rows(frame_idx: int, objects: Iterable) -> np.ndarray:
    """-> [frame_idx, grid_x float, grid_y float].

    ``ObjectEncoder.decode3d`` returns location=(x_cm, y_cm, 0) already
    relative to the VFA grid origin.  Divide by 10 cm; do NOT add/subtract the
    world origin again.
    """
    xy = object_xy_cm(objects)
    if len(xy) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    rows = np.zeros((len(xy), 3), dtype=np.float64)
    rows[:, 0] = float(frame_idx)
    rows[:, 1] = xy[:, 0] / EVAL_GRID_CELL_CM
    rows[:, 2] = xy[:, 1] / EVAL_GRID_CELL_CM
    return rows


def gt_rows(frame_idx: int, objects: Iterable) -> np.ndarray:
    """-> [frame_idx, rounded grid_x, rounded grid_y].

    Same centre convention as WorldTrack
    ``PedestrianDataset.prepare_gt(center_from_3d=True)``.
    """
    rows = prediction_rows(frame_idx, objects)
    if len(rows) == 0:
        return rows.astype(np.int64)
    rows[:, 1:] = np.rint(rows[:, 1:])
    return rows.astype(np.int64)


def object_confidences(objects: Iterable) -> np.ndarray:
    vals = []
    for obj in objects:
        if getattr(obj, "conf", None) is None:
            continue
        arr = np.asarray(to_numpy(obj.conf), dtype=np.float64).reshape(-1)
        if arr.size:
            vals.append(float(arr[0]))
    return np.asarray(vals, dtype=np.float64)


def check_coordinate_bounds(name: str, rows: np.ndarray,
                            world_size=EXPECTED_WORLD_SIZE) -> Dict[str, int]:
    if len(rows) == 0:
        return {"rows": 0, "out_of_bounds": 0}
    y_extent, x_extent = world_size
    bad = ((rows[:, 1] < 0.0) | (rows[:, 1] > x_extent / EVAL_GRID_CELL_CM) |
           (rows[:, 2] < 0.0) | (rows[:, 2] > y_extent / EVAL_GRID_CELL_CM))
    report = {"rows": int(len(rows)), "out_of_bounds": int(np.sum(bad))}
    if report["out_of_bounds"]:
        print(f"[COORD] WARNING: {name} has {report['out_of_bounds']}/"
              f"{report['rows']} rows outside the expected 10-cm grid.")
    return report


# -----------------------------------------------------------------------------
# Unit check against WorldTrack's GT convention
# -----------------------------------------------------------------------------
def expected_prepare_gt_rows(frame_ds, frame_idx: int) -> np.ndarray:
    """Recompute prepare_gt(center_from_3d=True) rows for one local frame."""
    adapter = frame_ds.base
    base3d = adapter.base3d
    frame_key = adapter.frame_list[frame_idx]
    boxes = base3d.get_3d_boxes(frame_key)
    rows = []
    for cow_id in sorted(boxes):
        center = boxes[cow_id]["center"]
        # PedestrianDataset.prepare_gt(center_from_3d=True):
        #   grid_x, grid_y = base.get_worldgrid_from_center(center)
        #   row = [frame, int(round(grid_x)), int(round(grid_y))]
        grid_x, grid_y = base3d.get_worldgrid_from_center(center)
        rows.append([frame_idx, int(round(float(grid_x))),
                     int(round(float(grid_y)))])
    if not rows:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


def comparable_gt_txt_rows(gt_txt: Path, frame_idx: int) -> np.ndarray | None:
    if not gt_txt.is_file() or gt_txt.stat().st_size == 0:
        return None
    arr = np.loadtxt(gt_txt, ndmin=2)
    arr = arr[np.rint(arr[:, 0]).astype(int) == int(frame_idx)]
    if len(arr) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    arr[:, 0] = np.rint(arr[:, 0])
    arr[:, 1:] = np.rint(arr[:, 1:])
    return arr.astype(np.int64)


def sorted_rows(rows: np.ndarray) -> np.ndarray:
    if len(rows) == 0:
        return rows.reshape(0, 3)
    return np.asarray(sorted(map(tuple, rows.tolist())), dtype=rows.dtype)


def unit_check_gt_writer(dataset: MultiSeqFrameDataset,
                         strict_gt_txt: bool = False) -> List[Dict]:
    """Verify the first annotated frame of every sequence before inference.

    The adapter's ``prepare_gt(center_from_3d=True)`` formula and the
    prediction-writer round-trip are fatal checks.  A sequence ``gt.txt``
    mismatch is diagnostic by default because the file can omit rows relative
    to the current 3D-centre source; pass ``--strict_gt_txt`` to make it fatal.
    """
    reports = []
    failures = []
    for seq_name, frame_ds in zip(dataset.sequence_names, dataset.datasets):
        labelled = [i for i, labels in enumerate(frame_ds.labels) if len(labels)]
        if not labelled:
            reports.append({"sequence": seq_name, "skipped": "no GT labels"})
            continue
        frame_idx = int(labelled[0])
        objects = frame_ds.labels[frame_idx]
        ours = gt_rows(frame_idx, objects)
        expected = expected_prepare_gt_rows(frame_ds, frame_idx)
        ok_expected = np.array_equal(sorted_rows(ours), sorted_rows(expected))

        # A det-writer orientation check: treating the same GT objects as
        # predictions must round to the same x/y rows.  This catches an x/y
        # swap in the writer before any model output is judged.
        det_style = prediction_rows(frame_idx, objects)
        det_rounded = det_style.copy()
        det_rounded[:, 1:] = np.rint(det_rounded[:, 1:])
        ok_writer = np.array_equal(
            sorted_rows(det_rounded.astype(np.int64)), sorted_rows(expected)
        )

        gt_txt = Path(frame_ds.root) / "gt.txt"
        txt_rows = comparable_gt_txt_rows(gt_txt, frame_idx)
        ok_txt = None
        if txt_rows is not None:
            ok_txt = np.array_equal(sorted_rows(ours), sorted_rows(txt_rows))

        report = {
            "sequence": seq_name,
            "frame_idx": frame_idx,
            "rows": int(len(ours)),
            "matches_prepare_gt_formula": bool(ok_expected),
            "prediction_writer_roundtrip": bool(ok_writer),
            "gt_txt": str(gt_txt) if txt_rows is not None else None,
            "matches_gt_txt": ok_txt,
        }
        if ok_txt is False:
            ours_set = set(map(tuple, ours.tolist()))
            txt_set = set(map(tuple, txt_rows.tolist()))
            report["missing_from_gt_txt"] = sorted(ours_set - txt_set)
            report["extra_in_gt_txt"] = sorted(txt_set - ours_set)
            report["ours"] = ours.tolist()
            report["expected"] = expected.tolist()
            report["gt_txt_rows"] = txt_rows.tolist()
        reports.append(report)

        if ok_txt is False:
            gt_status = "MISMATCH (warning)" if not strict_gt_txt else "FAIL"
        else:
            gt_status = "OK" if ok_txt else "not present"
        print(f"[UNIT] {seq_name} frame {frame_idx}: "
              f"prepare_gt formula {'OK' if ok_expected else 'FAIL'}, "
              f"det writer round-trip {'OK' if ok_writer else 'FAIL'}, "
              f"gt.txt {gt_status}")
        fatal = (not ok_expected or not ok_writer or
                 (strict_gt_txt and ok_txt is False))
        if fatal:
            failures.append(report)
    if failures:
        raise RuntimeError(
            "GT/det writer unit check failed. Fix the writer before "
            "interpreting model MODA:\n" + json.dumps(failures, indent=2)
        )
    return reports


# -----------------------------------------------------------------------------
# Prediction and MOD file writing
# -----------------------------------------------------------------------------
def threshold_values(args: argparse.Namespace) -> List[float]:
    if not args.sweep:
        return [float(args.cls_thresh)]
    n = int(round((args.sweep_max - args.sweep_min) / args.sweep_step)) + 1
    values = [args.sweep_min + i * args.sweep_step for i in range(n)]
    return [float(np.clip(round(v, 10), 0.0, 1.0)) for v in values]


def predict_rows_all_thresholds(model, dataset: MultiSeqFrameDataset,
                                encoder: ObjectEncoder,
                                thresholds: Sequence[float],
                                device: torch.device,
                                num_workers: int) -> Tuple[Dict[float, Dict[str, np.ndarray]], Dict[str, np.ndarray], Dict]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=num_workers, collate_fn=collate)
    det = {float(t): {name: [] for name in dataset.sequence_names}
           for t in thresholds}
    gt = {name: [] for name in dataset.sequence_names}
    bounds = {"prediction": {"rows": 0, "out_of_bounds": 0},
              "ground_truth": {"rows": 0, "out_of_bounds": 0}}

    model.eval()
    with torch.no_grad():
        with tqdm(loader, desc=f"[EVAL:{dataset.split}]", mininterval=1.0) as pbar:
            for global_idx, (_, images, objects, _, calibs, grid) in enumerate(pbar):
                seq_idx, seq_name, frame_idx = sequence_for_global_index(
                    dataset, global_idx)
                del seq_idx
                images = images.to(device, non_blocking=True)
                calibs = calibs.to(device, non_blocking=True)
                grid = grid.to(device, non_blocking=True)
                encoded_pred = model(images, calibs, grid)

                g_rows = gt_rows(frame_idx, objects[0])
                g_report = check_coordinate_bounds(
                    f"GT {seq_name}:{frame_idx}", g_rows)
                for key in bounds["ground_truth"]:
                    bounds["ground_truth"][key] += g_report[key]
                if len(g_rows):
                    gt[seq_name].append(g_rows)

                for threshold in thresholds:
                    preds = encoder.batch_decode(encoded_pred, float(threshold))
                    rows = prediction_rows(frame_idx, preds)
                    p_report = check_coordinate_bounds(
                        f"pred {seq_name}:{frame_idx}@{threshold:.2f}", rows)
                    bounds["prediction"]["rows"] += p_report["rows"]
                    bounds["prediction"]["out_of_bounds"] += p_report["out_of_bounds"]
                    if len(rows):
                        det[float(threshold)][seq_name].append(rows)

    det_np = {
        t: {name: (np.vstack(parts) if parts else np.zeros((0, 3)))
            for name, parts in by_seq.items()}
        for t, by_seq in det.items()
    }
    gt_np = {name: (np.vstack(parts) if parts else np.zeros((0, 3)))
             for name, parts in gt.items()}
    return det_np, gt_np, bounds


def write_rows(path: Path, rows: np.ndarray, prediction: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rows).reshape(-1, 3)
    if prediction:
        np.savetxt(path, arr, fmt=["%d", "%.6f", "%.6f"])
    else:
        np.savetxt(path, arr.astype(np.int64), fmt="%d")


def read_rows(path: Path) -> np.ndarray:
    if not path.is_file() or path.stat().st_size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.loadtxt(path, ndmin=2).astype(np.float64)


def write_sequence_files(out_dir: Path, sequence_names: Sequence[str],
                         det_by_seq: Dict[str, np.ndarray],
                         gt_by_seq: Dict[str, np.ndarray]) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    det_paths, gt_paths = {}, {}
    for seq in sequence_names:
        det_path = out_dir / f"det_{seq}.txt"
        gt_path = out_dir / f"gt_{seq}.txt"
        write_rows(det_path, det_by_seq.get(seq, np.zeros((0, 3))), True)
        write_rows(gt_path, gt_by_seq.get(seq, np.zeros((0, 3))), False)
        det_paths[seq] = det_path
        gt_paths[seq] = gt_path
    return det_paths, gt_paths


def combine_sequence_files(paths: Dict[str, Path], out_path: Path,
                           integer_rows: bool) -> np.ndarray:
    combined = []
    for seq_idx, seq in enumerate(paths):
        rows = read_rows(paths[seq])
        if len(rows) == 0:
            continue
        rows = rows.copy()
        rows[:, 0] += seq_idx * FRAME_SEQ_OFFSET
        combined.append(rows)
    arr = (np.vstack(combined) if combined else np.zeros((0, 3), dtype=np.float64))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if integer_rows:
        np.savetxt(out_path, arr.astype(np.int64), fmt="%d")
    else:
        np.savetxt(out_path, arr, fmt=["%d", "%.6f", "%.6f"])
    return arr


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def evaluate_one_threshold(threshold: float, run_dir: Path,
                           dataset: MultiSeqFrameDataset,
                           det_by_seq: Dict[str, np.ndarray],
                           gt_by_seq: Dict[str, np.ndarray],
                           mod_metrics) -> Dict:
    det_paths, gt_paths = write_sequence_files(
        run_dir, dataset.sequence_names, det_by_seq, gt_by_seq)
    all_det = combine_sequence_files(det_paths, run_dir / "det_all.txt", False)
    all_gt = combine_sequence_files(gt_paths, run_dir / "gt_all.txt", True)

    per_sequence = []
    for seq in dataset.sequence_names:
        n_gt = int(len(read_rows(gt_paths[seq])))
        n_det = int(len(read_rows(det_paths[seq])))
        if n_gt == 0:
            metrics = {"recall": None, "precision": None, "moda": None, "modp": None}
            print(f"[MOD] {seq}: skipped metric (no GT rows); det rows={n_det}")
        else:
            recall, precision, moda, modp = mod_metrics(
                str(det_paths[seq].resolve()), str(gt_paths[seq].resolve()),
                td_cells=EVAL_TD_CELLS)
            metrics = {"recall": float(recall), "precision": float(precision),
                       "moda": float(moda), "modp": float(modp)}
        per_sequence.append({"sequence": seq, "gt_rows": n_gt, "det_rows": n_det,
                             **metrics})

    if len(all_gt) == 0:
        raise RuntimeError("No GT rows were written for this split; cannot evaluate.")
    recall, precision, moda, modp = mod_metrics(
        str((run_dir / "det_all.txt").resolve()),
        str((run_dir / "gt_all.txt").resolve()),
        td_cells=EVAL_TD_CELLS)
    overall = {"recall": float(recall), "precision": float(precision),
               "moda": float(moda), "modp": float(modp),
               "gt_rows": int(len(all_gt)), "det_rows": int(len(all_det))}
    return {"threshold": float(threshold), "run_dir": str(run_dir),
            "overall": overall, "per_sequence": per_sequence,
            "det_paths": {k: str(v) for k, v in det_paths.items()},
            "gt_paths": {k: str(v) for k, v in gt_paths.items()}}


def print_metric_table(result: Dict) -> None:
    print("\n" + "-" * 86)
    print(f"POINT-MOD @ {EVAL_GATE_CM:.0f} cm | threshold {result['threshold']:.2f}")
    print("-" * 86)
    print(f"{'Sequence':<34}{'GT':>7}{'Det':>7}{'Recall':>10}"
          f"{'Precision':>11}{'MODA':>9}{'MODP':>9}")
    print("-" * 86)
    for row in result["per_sequence"]:
        if row["moda"] is None:
            print(f"{row['sequence']:<34}{row['gt_rows']:>7}{row['det_rows']:>7}"
                  f"{'skip':>10}{'skip':>11}{'skip':>9}{'skip':>9}")
        else:
            print(f"{row['sequence']:<34}{row['gt_rows']:>7}{row['det_rows']:>7}"
                  f"{row['recall']:>10.2f}{row['precision']:>11.2f}"
                  f"{row['moda']:>9.2f}{row['modp']:>9.2f}")
    overall = result["overall"]
    print("-" * 86)
    print(f"{'OVERALL':<34}{overall['gt_rows']:>7}{overall['det_rows']:>7}"
          f"{overall['recall']:>10.2f}{overall['precision']:>11.2f}"
          f"{overall['moda']:>9.2f}{overall['modp']:>9.2f}")
    print("-" * 86)


def print_sweep_table(results: Sequence[Dict], split: str) -> None:
    print("\n" + "=" * 74)
    print(f"{split.upper()} THRESHOLD SWEEP (overall sequences, 100 cm gate)")
    print("=" * 74)
    print(f"{'cls_thresh':>11}{'Recall':>10}{'Precision':>11}"
          f"{'MODA':>9}{'MODP':>9}{'Det':>8}")
    print("-" * 74)
    for result in results:
        overall = result["overall"]
        print(f"{result['threshold']:>11.2f}{overall['recall']:>10.2f}"
              f"{overall['precision']:>11.2f}{overall['moda']:>9.2f}"
              f"{overall['modp']:>9.2f}{overall['det_rows']:>8d}")
    print("-" * 74)


def write_sweep_csv(path: Path, results: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "threshold", "recall", "precision", "moda", "modp",
            "gt_rows", "det_rows", "run_dir"])
        writer.writeheader()
        for result in results:
            writer.writerow({
                "threshold": result["threshold"],
                "recall": result["overall"]["recall"],
                "precision": result["overall"]["precision"],
                "moda": result["overall"]["moda"],
                "modp": result["overall"]["modp"],
                "gt_rows": result["overall"]["gt_rows"],
                "det_rows": result["overall"]["det_rows"],
                "run_dir": result["run_dir"],
            })


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
def diagnostic_indices(dataset: MultiSeqFrameDataset, n: int) -> List[int]:
    """Choose up to n evenly spaced frames, preferring frames with GT."""
    candidates = []
    for (start, end), frame_ds in zip(dataset.sequence_boundaries, dataset.datasets):
        for local_idx in range(end - start):
            if len(frame_ds.labels[local_idx]):
                candidates.append(start + local_idx)
    if not candidates:
        candidates = list(range(len(dataset)))
    if len(candidates) <= n:
        return sorted(set(map(int, candidates)))
    picks = np.linspace(0, len(candidates) - 1, num=n)
    return sorted(set(int(candidates[int(round(p))]) for p in picks))


def save_heatmap_overlay(path: Path, gt_heatmap, pred_heatmap,
                         gt_objects, pred_objects) -> None:
    gt_map = np.asarray(to_numpy(gt_heatmap), dtype=np.float32)
    pred_map = np.asarray(to_numpy(pred_heatmap), dtype=np.float32)
    gt_xy = object_xy_cm(gt_objects) / EXPECTED_CUBE_LWH[0]
    pred_xy = object_xy_cm(pred_objects) / EXPECTED_CUBE_LWH[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, heatmap, title in zip(axes, (gt_map, pred_map), ("GT RGK", "Prediction")):
        im = ax.imshow(heatmap, cmap="viridis", vmin=0.0, vmax=1.0,
                       origin="upper", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("x cell (25 cm)")
        ax.set_ylabel("y cell (25 cm)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if len(gt_xy):
        for ax in axes:
            ax.scatter(gt_xy[:, 0], gt_xy[:, 1], s=55, facecolors="none",
                       edgecolors="lime", linewidths=1.5, label="GT")
    if len(pred_xy):
        axes[1].scatter(pred_xy[:, 0], pred_xy[:, 1], s=55, marker="x",
                        c="red", linewidths=1.5, label="pred")
    axes[1].legend(loc="upper right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_confidence_histogram(path: Path, confidences: np.ndarray,
                              threshold: float, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    if len(confidences):
        ax.hist(confidences, bins=30, range=(0.0, 1.0), color="steelblue",
                alpha=0.85)
        ax.axvline(threshold, color="red", linestyle="--",
                   label=f"threshold={threshold:.2f}")
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "no decoder candidates", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlim(0.0, 1.0)
    ax.set_yscale("log")
    ax.set_xlabel("post-NMS confidence")
    ax.set_ylabel("count (log)")
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def confidence_summary(confidences: np.ndarray) -> Dict[str, float | int | None]:
    nonzero = confidences[confidences > 1e-8]
    def f(func, arr):
        return None if len(arr) == 0 else float(func(arr))
    return {
        "candidates": int(len(confidences)),
        "nonzero_candidates": int(len(nonzero)),
        "mean_all": f(np.mean, confidences),
        "mean_nonzero": f(np.mean, nonzero),
        "median_nonzero": f(np.median, nonzero),
        "p10_nonzero": f(lambda x: np.percentile(x, 10), nonzero),
        "p90_nonzero": f(lambda x: np.percentile(x, 90), nonzero),
        "max": f(np.max, confidences),
    }


def save_projection_overlay(path: Path, image, calib, gt_objects,
                            pred_objects, title: str) -> None:
    fig = visualize_bboxes(image.detach().cpu(), calib.detach().cpu(),
                           gt_objects, pred_objects)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_diagnostics(model, dataset: MultiSeqFrameDataset,
                    encoder: ObjectEncoder, device: torch.device,
                    threshold: float, out_dir: Path,
                    n_frames: int, num_workers: int) -> Dict:
    """Decode n frames and save heatmap/projection/confidence diagnostics."""
    indices = set(diagnostic_indices(dataset, n_frames))
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=num_workers, collate_fn=collate)
    reports = []
    print("\n" + "=" * 86)
    print(f"DIAGNOSTICS: {len(indices)} frames at threshold {threshold:.2f}")
    print("=" * 86)
    model.eval()
    with torch.no_grad():
        for global_idx, (_, images, objects, heatmaps, calibs, grid) in enumerate(loader):
            if global_idx not in indices:
                continue
            _, seq_name, frame_idx = sequence_for_global_index(dataset, global_idx)
            images_dev = images.to(device)
            calibs_dev = calibs.to(device)
            grid_dev = grid.to(device)
            encoded_pred = model(images_dev, calibs_dev, grid_dev)

            # cls_thresh=-1 exposes the whole post-NMS top-k candidate set,
            # including low-confidence candidates.  This is the histogram that
            # distinguishes a bad threshold from an absent signal.
            candidate_objects = encoder.batch_decode(encoded_pred, -1.0)
            pred_objects = encoder.batch_decode(encoded_pred, threshold)
            conf = object_confidences(candidate_objects)
            summary = confidence_summary(conf)
            gt_count = len(objects[0])
            pred_count = len(pred_objects)
            print(
                f"[DIAG] {seq_name} frame {frame_idx}: GT={gt_count}, "
                f"detections={pred_count}, candidates={summary['candidates']} "
                f"(nonzero={summary['nonzero_candidates']}), "
                f"conf mean={summary['mean_all'] if summary['mean_all'] is not None else float('nan'):.4f}, "
                f"mean nonzero={summary['mean_nonzero'] if summary['mean_nonzero'] is not None else float('nan'):.4f}, "
                f"p10/p90 nonzero="
                f"{summary['p10_nonzero'] if summary['p10_nonzero'] is not None else float('nan'):.4f}/"
                f"{summary['p90_nonzero'] if summary['p90_nonzero'] is not None else float('nan'):.4f}, "
                f"max={summary['max'] if summary['max'] is not None else float('nan'):.4f}"
            )

            stem = f"{safe_tag(seq_name)}_frame{frame_idx:05d}"
            heatmap_path = out_dir / f"{stem}_heatmap.png"
            projection_path = out_dir / f"{stem}_projection_cam1.png"
            hist_path = out_dir / f"{stem}_confidence_hist.png"
            save_heatmap_overlay(
                heatmap_path, heatmaps[0],
                torch.sigmoid(encoded_pred["heatmap"])[0, 0],
                objects[0], pred_objects)
            save_projection_overlay(
                projection_path, images[0], calibs[0], objects[0], pred_objects,
                f"{seq_name} frame {frame_idx} | GT {gt_count} | pred {pred_count}")
            save_confidence_histogram(
                hist_path, conf, threshold,
                f"{seq_name} frame {frame_idx} confidence")
            reports.append({
                "global_index": int(global_idx),
                "sequence": seq_name,
                "frame_idx": int(frame_idx),
                "gt_count": int(gt_count),
                "detections": int(pred_count),
                "confidence": summary,
                "heatmap": str(heatmap_path),
                "projection_cam1": str(projection_path),
                "confidence_histogram": str(hist_path),
            })
            if len(reports) == len(indices):
                break

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "diagnostics.json"
    with summary_path.open("w") as f:
        json.dump(json_safe({"threshold": threshold, "frames": reports}), f, indent=2)
    print(f"[DIAG] Saved overlays, histograms and summary to {out_dir}")
    print(f"[DIAG] Summary: {summary_path}")
    return {"threshold": threshold, "frames": reports, "summary": str(summary_path)}


# -----------------------------------------------------------------------------
# Main evaluation flow
# -----------------------------------------------------------------------------
def run() -> Dict:
    args = parse_args()
    args.worldtrack_root = resolve_worldtrack_root(args.worldtrack_root)
    mod_metrics = import_mod_metrics(args.worldtrack_root)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    ck_args, ckpt_payload = checkpoint_metadata(checkpoint)
    checkpoint_epoch = ckpt_payload.get("epoch")
    del ckpt_payload
    topk = int(args.topk if args.topk is not None else getattr(ck_args, "topk", 100))
    resize = tuple(args.resize_size or getattr(ck_args, "resize_size",
                                                mmcows_opts.resize_size))
    run_tag = safe_tag(args.run_tag or checkpoint.stem)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print("Settings:")
    print(json.dumps(json_safe(vars(args)), indent=2))
    print(f"Resolved checkpoint: {checkpoint}")
    print(f"Checkpoint epoch: {checkpoint_epoch}")
    print(f"Resolved WorldTrack root: {args.worldtrack_root}")
    print(f"Device: {device}; decoder top-k: {topk}; resize: {resize}")

    dataset = build_eval_dataset(args, resize)
    eval_split = dataset.split
    verify_mmcows_geometry(dataset)
    unit_reports = ([] if args.skip_unit_check else
                    unit_check_gt_writer(dataset, strict_gt_txt=args.strict_gt_txt))

    model = resume_stage6_model(checkpoint, device)
    # The encoder only needs shared geometry/class metadata.  Passing the first
    # per-sequence frameDataset also works with the pre-multiseq ObjectEncoder.
    encoder = ObjectEncoder(first_frame_dataset(dataset), topk=topk)
    if not hasattr(encoder, "_location_to_grid"):
        print("[ENCODER] WARNING: this ObjectEncoder predates the MmCows "
              "non-square-grid encode fix. Decoding remains geometrically "
              "consistent, but a model trained with that encoder may have "
              "location/dimension/rotation supervision at transposed cells. "
              "Check vfa/data/encoder.py before retraining.")
    thresholds = threshold_values(args)
    print(f"Decode threshold(s): {', '.join(f'{t:.2f}' for t in thresholds)}")

    det_rows, gt_rows_by_seq, bounds = predict_rows_all_thresholds(
        model, dataset, encoder, thresholds, device, args.num_workers)
    print(f"[COORD] Ground-truth rows: {bounds['ground_truth']}; "
          f"prediction rows across all thresholds: {bounds['prediction']}")

    results = []
    for threshold in thresholds:
        if args.sweep:
            run_dir = (output_dir / "sweep" / eval_split / run_tag /
                       f"thr_{threshold:.2f}")
        else:
            # Stage-7 canonical path: experiments_mmcows/eval/det_<seq>.txt.
            run_dir = output_dir
        result = evaluate_one_threshold(
            threshold, run_dir, dataset, det_rows[float(threshold)],
            gt_rows_by_seq, mod_metrics)
        results.append(result)
        if not args.sweep:
            print_metric_table(result)

    if args.sweep:
        print_sweep_table(results, eval_split)
        best = max(results, key=lambda r: (
            r["overall"]["moda"], r["overall"]["modp"], r["overall"]["recall"]))
        sweep_dir = output_dir / "sweep" / eval_split / run_tag
        sweep_csv = sweep_dir / "sweep_summary.csv"
        write_sweep_csv(sweep_csv, results)
        print(f"Sweep CSV: {sweep_csv}")
        print(f"Best {eval_split} threshold by MODA: {best['threshold']:.2f} "
              f"(MODA={best['overall']['moda']:.2f}, "
              f"MODP={best['overall']['modp']:.2f}, "
              f"precision={best['overall']['precision']:.2f}, "
              f"recall={best['overall']['recall']:.2f})")
        print_metric_table(best)

        # Canonical files at experiments_mmcows/eval/det_<seq>.txt correspond
        # to the sweep winner for convenient inspection and test-time parity.
        best_dir = Path(best["run_dir"])
        for seq in dataset.sequence_names:
            shutil.copyfile(best_dir / f"det_{seq}.txt", output_dir / f"det_{seq}.txt")
            shutil.copyfile(best_dir / f"gt_{seq}.txt", output_dir / f"gt_{seq}.txt")
        shutil.copyfile(best_dir / "det_all.txt", output_dir / "det_all.txt")
        shutil.copyfile(best_dir / "gt_all.txt", output_dir / "gt_all.txt")
        result = best
    else:
        result = results[0]

    summary = {
        "stage": "Stage 7 - BEVine3D point-MOD evaluation",
        "data": args.data,
        "split": eval_split,
        "requested_split": args.split,
        "sequence": args.sequence,
        "root": os.path.abspath(args.root),
        "manifest": args.manifest,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "run_tag": run_tag,
        "threshold": result["threshold"],
        "thresholds": thresholds,
        "evaluator": "WorldTrack evaluation.mod.modMetricsCalculator",
        "td_cells": EVAL_TD_CELLS,
        "gate_cm": EVAL_GATE_CM,
        "grid_cell_cm": EVAL_GRID_CELL_CM,
        "topk": topk,
        "resize_size": resize,
        "unit_checks": unit_reports,
        "coordinate_bounds": bounds,
        "result": result,
        "sweep_results": results if args.sweep else None,
        "canonical_det_gt_dir": str(output_dir.resolve()),
    }

    diagnostics_needed = (args.diagnostics or
                          result["overall"]["moda"] <= args.diagnostic_moda)
    if diagnostics_needed:
        diag_dir = output_dir / "diagnostics" / eval_split / run_tag / f"thr_{result['threshold']:.2f}"
        print(f"[DIAG] Triggered: overall MODA={result['overall']['moda']:.2f}; "
              f"threshold={result['threshold']:.2f}")
        summary["diagnostics"] = run_diagnostics(
            model, dataset, encoder, device, result["threshold"], diag_dir,
            args.diagnose_frames, args.num_workers)
    else:
        summary["diagnostics"] = None

    summary_name = (f"evaluation_{eval_split}_{run_tag}"
                    f"_thr{result['threshold']:.2f}.json")
    summary_path = output_dir / summary_name
    with summary_path.open("w") as f:
        json.dump(json_safe(summary), f, indent=2)

    overall = result["overall"]
    print("\nPaper-row numbers (percent, 100 cm gate):")
    print("| Method | Split | MODA | MODP | Precision | Recall |")
    print("|---|---:|---:|---:|---:|---:|")
    print(f"| VFA | {eval_split} | {overall['moda']:.2f} | "
          f"{overall['modp']:.2f} | {overall['precision']:.2f} | "
          f"{overall['recall']:.2f} |")
    print("\nMethods footnote:")
    print("VFA uses a ResNet-18 ImageNet backbone; only mmcows_test_only is "
          "evaluated; its sequence and 3D supervision centres are identical "
          "to BEVine3D; centres are exported on the 10-cm MmCows grid; the "
          "association gate is 100 cm; metrics come from WorldTrack "
          "evaluation.mod.modMetricsCalculator.")
    print(f"\nSummary JSON: {summary_path}")
    print(f"Canonical det/gt files: {output_dir.resolve()}")
    return summary


if __name__ == "__main__":
    run()
