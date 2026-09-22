# vfa/data/multiseq_dataset.py
"""Multi-sequence wrapper for the VFA MmCows adapter + BEVine3D parity filter.

Stage 4 of the VFA-on-MmCows adaptation.  Three pieces:

1. ``MultiSeqFrameDataset`` — concatenates per-sequence
   ``vfa.data.dataset.frameDataset`` objects (each built with
   ``split='all'``), addressing them with ``bisect`` on cumulative sizes
   exactly like BEVine3D's ``datasets/multiseq_pedestrian_dataset.py``
   (``MultiSeqPedestrianDataset``).  ``__getitem__`` returns the
   sub-dataset's UNCHANGED 6-tuple
   ``(index, images, objects, heatmaps, calibs, grid)``, so
   ``vfa/utils.py::collate`` and ``vfa/trainer.py`` work untouched (the
   trainer never reads ``index``; collate only stacks).  ``index`` is the
   PER-SEQUENCE positional index, same convention as BEVine3D, whose
   samples also carry their local frame index.

2. ``build_multiseq(root, manifest_name, split, transform)`` — reads the
   split manifest (``sequences_mmcows_25percent.json``) and instantiates
   the MmCows adapter + one frameDataset per sequence of the requested
   split ('train' | 'val' | 'test'), skipping missing sequence
   directories with a warning — the same policy as BEVine3D's
   ``PedestrianDataModule._create_multiseq_dataset``.

3. Stationary-frame parity filter — a faithful reimplementation of the
   filter configured in WorldTrack's ``configs/d_mmcows_multiseq.yml``
   (``drop_stationary: true``, ``stationary_min_disp_cm: 5.0``,
   ``stationary_keep_prob: 0.1``, ``stationary_max_run: 5``), measuring
   displacements on metric 3D centres (``MmCows3D.get_center_overrides``,
   i.e. 3D box centroids merged with the centers_3d.json cache).
   DEFAULT ON for the train split; val/test are NEVER filtered (hard
   guard in ``build_multiseq`` — passing ``drop_stationary=True`` for
   val/test is ignored with a loud warning).

   Decision logic is IDENTICAL to BEVine3D's
   ``PedestrianDataset._filter_stationary_frames``:
     * frame 0 is always kept;
     * a frame is kept whenever ANY cow id shared with the immediately
       preceding frame moved more than ``min_disp_cm`` (metric cm);
     * a frame whose displacement cannot be measured (no metric centres,
       or no shared cow ids) is kept — "cannot be proven stationary";
     * an all-stationary frame is dropped with probability
       ``1 - keep_prob``, but never more than ``max_run`` drops in a row
       (the coin is only drawn when the run cap has not been hit, so the
       draw ORDER also matches BEVine3D).

   Two deliberate differences, both demanded by the Stage-4 spec:
     * the coin flips come from a NUMPY generator
       (``np.random.default_rng(STATIONARY_SEED)``) instead of
       ``random.Random(STATIONARY_SEED)``.  Same seed constant, same
       draw pattern, but a different random stream: the two pipelines
       drop slightly different frames, so
       ``tests/test_multiseq_parity.py`` compares train counts within
       the binomial tolerance of the keep/drop draw, not exactly.  The
       STATIONARY-FRAME IDENTIFICATION itself is deterministic and must
       match BEVine3D exactly (asserted in that test).
     * no positionID fallback: frames without metric centres are kept
       instead of being measured on the 10 cm-quantised positionID
       decode.  On the 25% manifest every frame has metric centres
       (MmCows3D verifies the cache against the 3D boxes at load time),
       so this changes nothing in practice; the report counts such
       frames so a coverage regression is visible.
"""

import bisect
import json
import os

import numpy as np
import torch

from vfa.data.dataset import frameDataset
from vfa.data.mmcows import MmCows as VfaMmCows, SPLIT_MANIFEST_NAME

__all__ = ['MultiSeqFrameDataset', 'build_multiseq', 'stationary_frame_mask',
           'DROP_STATIONARY', 'STATIONARY_MIN_DISP_CM', 'STATIONARY_KEEP_PROB',
           'STATIONARY_MAX_RUN', 'STATIONARY_SEED']

# Stationary-filter defaults — mirror configs/d_mmcows_multiseq.yml.
# Master default for build_multiseq(): False = filter OFF unless the
# caller explicitly passes drop_stationary=True (train split only).
DROP_STATIONARY = False
STATIONARY_MIN_DISP_CM = 5.0
STATIONARY_KEEP_PROB = 0.1
STATIONARY_MAX_RUN = 5
# Same seed constant as BEVine3D's filter (random.Random(20240717)),
# feeding a numpy generator instead (see module docstring).
STATIONARY_SEED = 20240717


class MultiSeqFrameDataset(torch.utils.data.Dataset):
    """Concatenate per-sequence frameDataset objects (one per sequence).

    Index arithmetic mirrors BEVine3D's MultiSeqPedestrianDataset:
    ``bisect_right`` on cumulative sizes, ``sequence_boundaries`` as
    (start, end) index pairs, ``is_multi_seq`` marker property.
    """

    def __init__(self, datasets, sequence_names=None):
        super().__init__()
        if not datasets:
            raise ValueError('MultiSeqFrameDataset needs at least one '
                             'sub-dataset (all sequence dirs missing?)')
        self.datasets = list(datasets)
        self.sequence_names = (
            list(sequence_names) if sequence_names is not None else
            [os.path.basename(os.path.normpath(ds.root))
             for ds in self.datasets])
        if len(self.sequence_names) != len(self.datasets):
            raise ValueError('sequence_names and datasets differ in length')

        self.cumulative_sizes = []
        self.sequence_boundaries = []
        cumsum = 0
        for ds in self.datasets:
            start = cumsum
            cumsum += len(ds)
            self.sequence_boundaries.append((start, cumsum))
            self.cumulative_sizes.append(cumsum)

        # Per-sequence stationary-filter reports (train only; None for
        # val/test).  Attached to the frameDatasets by build_multiseq.
        self.stationary_reports = [getattr(ds, 'stationary_report', None)
                                   for ds in self.datasets]

    # ------------------------------------------------------------------
    # Core Dataset interface
    # ------------------------------------------------------------------
    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if not 0 <= idx < len(self):
            raise IndexError(f'MultiSeqFrameDataset index {idx} out of '
                             f'range [0, {len(self)})')
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        # UNCHANGED 6-tuple (index, images, objects, heatmaps, calibs,
        # grid) from the sub-dataset — vfa.utils.collate and
        # vfa.trainer.Trainer work untouched.
        return self.datasets[dataset_idx][sample_idx]

    # ------------------------------------------------------------------
    # Helpers (parity with BEVine3D's MultiSeqPedestrianDataset)
    # ------------------------------------------------------------------
    def get_sequence_boundaries(self):
        """Return list of (start, end) index pairs, one per sequence."""
        return self.sequence_boundaries

    @property
    def is_multi_seq(self):
        return True


# ----------------------------------------------------------------------
# Stationary-frame filter (train split only) — BEVine3D parity
# ----------------------------------------------------------------------
def stationary_frame_mask(base3d,
                          min_disp_cm=STATIONARY_MIN_DISP_CM,
                          keep_prob=STATIONARY_KEEP_PROB,
                          max_run=STATIONARY_MAX_RUN,
                          seed=STATIONARY_SEED):
    """Decide which frames of one sequence survive the stationary filter.

    ``base3d`` is the wrapped ``MmCows3D`` of the sequence (metric 3D
    centres via ``get_center_overrides``).  Returns
    ``(kept_positions, report)`` where ``kept_positions`` is the sorted
    list of POSITIONAL indices into ``base3d.frame_list`` that survive.

    Logic mirrors PedestrianDataset._filter_stationary_frames (see the
    module docstring for the two deliberate differences).
    """
    min_disp = float(min_disp_cm)
    keep_prob = float(keep_prob)
    max_run = int(max_run)
    rng = np.random.default_rng(seed)   # fixed seed -> reproducible

    keys = list(base3d.frame_list)
    centres, n_cache = {}, 0
    for k in keys:
        c = base3d.get_center_overrides(k)
        if c:
            centres[k] = {int(i): (float(v[0]), float(v[1]))
                          for i, v in c.items()}
            n_cache += 1
    n_no_metric = len(keys) - n_cache

    kept, dropped = [], []
    run = 0                     # consecutive dropped stationary frames
    n_stationary = 0
    for i, k in enumerate(keys):
        if i == 0:
            kept.append(i)      # first frame always kept (BEVine3D too)
            continue
        prev, cur = centres.get(keys[i - 1], {}), centres.get(k, {})
        shared = set(prev) & set(cur)
        max_d = max(
            (float(np.hypot(cur[p][0] - prev[p][0],
                            cur[p][1] - prev[p][1])) for p in shared),
            default=None,
        )
        # Moved (or cannot be proven stationary) -> ALWAYS keep.
        if max_d is None or max_d > min_disp:
            kept.append(i)
            run = 0
            continue
        n_stationary += 1
        # All cows stationary -> drop with prob (1 - keep_prob), but
        # never more than max_run in a row.  The draw happens ONLY when
        # the run cap has not been hit — same short-circuit as
        # BEVine3D's `run < max_run and rng.random() >= keep_prob`, so
        # the draw sequence is aligned 1:1 with BEVine3D's.
        if run < max_run and rng.random() >= keep_prob:
            dropped.append(i)
            run += 1
        else:
            kept.append(i)
            run = 0

    report = {
        'sequence': os.path.basename(os.path.normpath(str(base3d.root))),
        'frames_in_split': len(keys),
        'stationary': n_stationary,
        'dropped': len(dropped),
        'kept': len(kept),
        'min_disp_cm': min_disp,
        'keep_prob': keep_prob,
        'max_drop_run': max_run,
        'seed': int(seed),
        'rng': 'numpy.default_rng',
        'metric_centre_frames': n_cache,
        'frames_without_metric_centres': n_no_metric,
        'dropped_keys': [str(keys[i]) for i in dropped],
        'dropped_positions': [int(i) for i in dropped],
    }
    return kept, report


def _apply_stationary_filter(frame_ds, kept_positions, report):
    """Subset a split='all' frameDataset IN PLACE to ``kept_positions``.

    frameDataset.__getitem__ uses ``frame_range[index]`` for image paths
    and ``labels[index]`` / ``heatmaps[index]`` for supervision, so all
    three are subset in parallel.  ``fpaths`` stays keyed by the ORIGINAL
    positional index — lookups go through the (already subset)
    ``frame_range``, so no change is needed there.
    """
    kept = [int(i) for i in kept_positions]
    frame_ds.frame_range = [frame_ds.frame_range[i] for i in kept]
    frame_ds.labels = [frame_ds.labels[i] for i in kept]
    hm = frame_ds.heatmaps
    if isinstance(hm, np.ndarray):
        frame_ds.heatmaps = hm[kept]          # fancy indexing -> copy
    else:
        frame_ds.heatmaps = [hm[i] for i in kept]
    assert len(frame_ds.frame_range) == len(frame_ds.labels) \
        == len(frame_ds.heatmaps), \
        'stationary filter broke frame_range/labels/heatmaps alignment'
    frame_ds.stationary_report = report
    return frame_ds


# ----------------------------------------------------------------------
# Construction helper
# ----------------------------------------------------------------------
def build_multiseq(root,                     # MmCows DATA root (multi-seq)
                   manifest_name=SPLIT_MANIFEST_NAME,
                   split='train',
                   transform=None,
                   # stationary parity flag: default OFF everywhere;
                   # FORCED OFF for val/test (they are never filtered)
                   drop_stationary=DROP_STATIONARY,
                   stationary_min_disp_cm=STATIONARY_MIN_DISP_CM,
                   stationary_keep_prob=STATIONARY_KEEP_PROB,
                   stationary_max_run=STATIONARY_MAX_RUN,
                   stationary_seed=STATIONARY_SEED,
                   annotation_mode='3d',
                   worldtrack_root=None,
                   reload_heatmaps=False):
    """Build the multi-sequence VFA dataset for one manifest split.

    Returns a ``MultiSeqFrameDataset`` whose sub-datasets are
    ``frameDataset(VfaMmCows(seq_dir), split='all')`` in MANIFEST ORDER,
    with the stationary filter applied per sequence when (and only when)
    ``split == 'train'`` and ``drop_stationary=True`` is passed
    explicitly (default: OFF — see DROP_STATIONARY).
    """
    assert split in ('train', 'val', 'test'), \
        f"split must be 'train'/'val'/'test', got {split!r}"

    if drop_stationary and split != 'train':
        print(f"  MultiSeq(VFA): ⚠ drop_stationary=True requested for "
              f"split {split!r} — IGNORED: val/test are NEVER filtered.")
        drop_stationary = False

    manifest_path = os.path.join(root, manifest_name)
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    if split not in manifest:
        raise ValueError(f"Split {split!r} not in {manifest_path}. "
                         f"Available: {list(manifest.keys())}")

    datasets, sequence_names = [], []
    tot_before, tot_after = 0, 0
    for seq_name in manifest[split]:
        seq_dir = os.path.join(root, str(seq_name))
        if not os.path.isdir(seq_dir):
            print(f"WARNING: Sequence dir not found: {seq_dir}, skipping.")
            continue

        adapter = VfaMmCows(seq_dir,
                            worldtrack_root=worldtrack_root,
                            annotation_mode=annotation_mode,
                            reload_heatmaps=reload_heatmaps)
        # transform=None -> frameDataset's own default (ToTensor())
        fds = (frameDataset(adapter, transform=transform, split='all')
               if transform is not None else
               frameDataset(adapter, split='all'))

        n_all = len(fds)
        if drop_stationary:
            kept, report = stationary_frame_mask(
                adapter.base3d,
                min_disp_cm=stationary_min_disp_cm,
                keep_prob=stationary_keep_prob,
                max_run=stationary_max_run,
                seed=stationary_seed)
            fds = _apply_stationary_filter(fds, kept, report)
            print(f"  [STATIONARY] {seq_name}: "
                  f"{report['stationary']}/{report['frames_in_split']} "
                  f"frames fully stationary (no cow moved > "
                  f"{report['min_disp_cm']:g} cm) -> dropped "
                  f"{report['dropped']}, kept {report['kept']} "
                  f"(metric centres: {report['metric_centre_frames']}, "
                  f"no-metric-centre frames: "
                  f"{report['frames_without_metric_centres']})")
            if report['frames_without_metric_centres']:
                print(f"  [STATIONARY] ⚠ {seq_name}: "
                      f"{report['frames_without_metric_centres']} frames "
                      f"had NO metric centres and were kept by default "
                      f"(BEVine3D would measure them on positionID).")
        else:
            fds.stationary_report = None

        print(f"  Seq {len(datasets)}: {seq_name} -> {len(fds)} frames"
              + (f" (of {n_all})" if len(fds) != n_all else ""))
        tot_before += n_all
        tot_after += len(fds)
        datasets.append(fds)
        sequence_names.append(str(seq_name))

    if not datasets:
        raise ValueError(f"No valid sequences for split {split!r} "
                         f"(manifest {manifest_path})")

    multi = MultiSeqFrameDataset(datasets, sequence_names=sequence_names)
    multi.split = split
    multi.manifest_path = manifest_path
    if drop_stationary:
        reps = [r for r in multi.stationary_reports if r]
        tot_s = sum(r['stationary'] for r in reps)
        tot_d = sum(r['dropped'] for r in reps)
        print(f"  [STATIONARY] split '{split}': {tot_s}/{tot_before} "
              f"frames fully stationary; dropped {tot_d} "
              f"({100.0 * tot_d / max(tot_before, 1):.1f}% of split)")
    print(f"  Total {split} frames: {len(multi)}"
          + (f" (unfiltered: {tot_before})"
             if len(multi) != tot_before else ""))
    return multi
