# vfa/data/mmcows.py
"""VFA adapter for the MmCows dataset (multi-sequence, 4 static cameras).

Wraps BEVine3D/WorldTrack's ``datasets.mmcows3d_dataset.MmCows3D`` so that
calibration loading, frame lists, 3D centroids and 3D boxes are IDENTICAL
to what BEVine3D sees.  Only the *interface* is translated to what
``vfa.data.dataset.frameDataset`` and ``vfa.data.encoder.ObjectEncoder``
expect from a base dataset (cf. ``vfa.data.multiviewC.MultiviewC``).

FIXED GEOMETRY (decided in Stage 1 — do not reinvent):
  * metric world: x in [-879, 1042] cm, y in [-646, 533] cm
  * VFA world frame (make_grid order: length = y-extent, width = x-extent):
        world_size = (1200, 1925) cm, origin at (x+879, y+646) = (0, 0)
  * cube_LWH = (25, 25, 32) cm, grid_h = 160
        -> BEV grid 48 (y) x 77 (x) cells = reduced_grid_size [48, 77]
  * stored Obj3D.location = (x_cm + 879, y_cm + 646, 0.0)   [VFA frame, cm]
        Obj3D.dimension = [h, w, l] cm      (VFA convention!)
        Obj3D.rotation  = yaw in radians    (see YAW CONVENTION below)
  * the inverse shift for camera projection / model input is
        x_cm = location[0] - 879 ; y_cm = location[1] - 646
    (mmcows_convert(grid): grid[...,0] += -879.0; grid[...,1] += -646.0)
  * images 2800x4480 (HxW) are resized to (700, 1120); intrinsics are
    pre-scaled here:  K' = diag(0.25, 0.25, 1) @ K.

YAW CONVENTION (Stage-2 projection gate):
  BEVine3D stores OBBs as (center, length, width, height, yaw) with yaw in
  RADIANS, CCW positive in the BEV plane, length along the box's local +x
  (WorldTrack evaluation/obb_metrics.py header).  VFA's
  ``compute_3d_bbox`` builds the cuboid with l along local +x, w along
  local +y, then applies rotz(rotation) (CCW).  Whether the two frames
  agree exactly, or are mirrored (image y-down vs world y-up), is decided
  EMPIRICALLY by tests/test_mmcows_projection.py, which sweeps the four
  candidate mappings in YAW_MODES and picks the one with the highest
  median envelope-vs-2D-box IoU.

    >>> RESULT (mmcows_1_train, frames 0/450/900, 149 cow-camera pairs
    >>> with a 2D annotation):  winner = 'raw' — median envelope-vs-2D IoU
    >>> 0.5185, margin +0.0323 over the best distinct runner-up 'neg'
    >>> (0.4863); 'raw90swap' ties 'raw' exactly (same rectangle).
    >>> 'raw' is also geometrically consistent with BEVine3D's documented
    >>> OBB convention (yaw rad, CCW positive, length along local +x).

  NOTE: 'raw' and 'raw90swap' are geometrically identical rectangle
  mappings (rotating a rectangle by +90 deg while swapping length/width
  yields the same corners), so their medians must tie — the sweep's real
  discriminators are 'neg' (mirrored yaw) and 'negswap' (mirrored yaw
  with l/w exchanged, a DIFFERENT, ~90-deg-rotated footprint for
  non-square cows).  All four are still evaluated and printed as
  specified.

Z-AXIS CONVENTION (Stage-2 diagnostic):
  The recovered extrinsics (Homography/PnP from floor-plane
  correspondences) can live in a right-handed world frame whose +z axis
  points INTO the floor (r3 = r1 x r2 completion when the (x, y)
  annotation frame is left-handed as seen by the camera).  Symptoms:
  cam_center z < 0 although the cameras are ~4 m above the floor, and
  projected cuboids extending BELOW the cows' feet while the z=0 grid
  projects correctly.  If tests/test_mmcows_projection.py picks z-mode
  'down', set WORLD_Z_SIGN = -1: this negates the z-column of [R|t],
  leaving the z=0 floor projection UNCHANGED while every cuboid/pillar
  built at z in [0, +h] (compute_3d_bbox, VFA's voxel pillars) extends
  PHYSICALLY upward.  x, y, yaw and the intrinsics are untouched.

    >>> RESULT (mmcows_1_train, frames 0/450/900, 149 cow-camera pairs):
    >>> z-mode 'down' won decisively — median IoU 0.5185 vs 0.0786
    >>> ('up') and 0.2605 ('center'); all four camera centres sit at
    >>> z in [-482, -347] cm, i.e. BELOW the z=0 floor in a +z-up
    >>> reading.  WORLD_Z_SIGN = -1 is therefore the permanent setting
    >>> below.

Stage 3 (this file's download()): per-frame RGK heatmaps are built per
sequence and cached at vfa/data/mmcows_RGK_world<LxW>_cube<LxWxH>_<seq>.npy
(per-sequence AND grid-geometry path — MultiviewC's fixed path would collide
across sequences, and a sequence-only name would silently reuse heatmaps from
a different world_size/cube_LWH setting); the classAverage ('Cow', [h,w,l] cm)
is aggregated over the TRAIN sequences of the split manifest ONCE and cached
at vfa/data/mmcows_ClAvg.json; all splits load that file, so val/test never
contribute to the mean.
"""

import json
import os
import sys

import numpy as np
from torchvision.datasets.vision import VisionDataset

from vfa.data.RGK import RotationGaussianKernel
from vfa.data.ClsAvg import ClassAverage
from vfa.utils import Obj3D

# RGK.py is VFA-original (frozen) and calls np.int, removed in NumPy >= 1.24.
# Shim it here, before any gaussian_kernel_heatmap() call, instead of editing
# RGK.py.  (Same shim as tests/test_mmcows_projection.py.)
if not hasattr(np, 'int'):
    np.int = int

MMCOWS_BBOX_LABEL_NAMES = ['Cow']

# Split manifest inside the MmCows data root (same file BEVine3D reads).
SPLIT_MANIFEST_NAME = 'sequences_mmcows_25percent.json'

# Cache directory for the per-sequence RGK heatmaps and the class-average
# file.  Resolved absolutely from this file so the adapter works no matter
# which CWD the training / test script is launched from (MultiviewC uses the
# relative path r'vfa/data/...', which only works from the repo root).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
MMCOWS_CACHE_DIR = os.path.join(REPO_ROOT, 'vfa', 'data')

# MmCows metric world bounds (cm), from WorldTrack datasets/mmcows_dataset.py
WORLD_X_MIN_CM = -879.0
WORLD_Y_MIN_CM = -646.0

# native image 2800x4480 (HxW) -> resized 700x1120
IMG_SCALE = 0.25
INTRINSIC_SCALE = np.diag([IMG_SCALE, IMG_SCALE, 1.0])

# Candidate yaw/length-width conventions evaluated by the Stage-2 test.
YAW_MODES = ('raw', 'neg', 'raw90swap', 'negswap')

# Selected convention (module-level so the diagnostic test can override it
# and rebuild the adapter).  MUST be set to the Stage-2 winner once the
# projection gate has run on a real sequence.
YAW_MODE = 'raw'

# +1: world +z points up (corners 0..+h are above the floor).
# -1: world +z points INTO the floor -> negate the z-column of [R|t] so
#     VFA's z in [0, +h] volume is physically above ground.
# Confirmed by the Stage-2 z-mode sweep on mmcows_1_train (z-mode 'down'
# won, median IoU 0.5185 vs 0.0786 for 'up'); see module docstring.
WORLD_Z_SIGN = -1

def rgk_cache_filename(seq_name, world_size, cube_LWH):
    """Geometry-qualified RGK cache filename.

    The RGK array shape and kernel placement are a function of BOTH the
    sequence and the BEV discretisation.  A sequence-only cache name is
    unsafe when world_size or cube_LWH changes (a stale 48x77 cache would
    otherwise be loaded for a different grid), so both are embedded here.
    """
    world_tag = 'x'.join(str(int(v)) for v in world_size)
    cube_tag = 'x'.join(str(int(v)) for v in cube_LWH)
    return f'mmcows_RGK_world{world_tag}_cube{cube_tag}_{seq_name}.npy'

def apply_yaw_convention(yaw, length, width, mode):
    """Map a stored (yaw, length, width) triple to VFA's (rotation, l, w).

    VFA's compute_3d_bbox: l along local +x, w along local +y, rotz CCW.
    Modes:
      'raw'       : rotation = yaw,        l/w unchanged
      'neg'       : rotation = -yaw,       l/w unchanged   (mirrored)
      'raw90swap' : rotation = yaw + pi/2, l<->w swapped   (== 'raw')
      'negswap'   : rotation = -yaw,       l<->w swapped   (DISTINCT from
                    'neg': swapping l/w without a +90 deg yaw change
                    rotates the footprint of a non-square box)
    Returns (rotation, length, width) — note the return order.
    """
    if mode == 'raw':
        return yaw, length, width
    if mode == 'neg':
        return -yaw, length, width
    if mode == 'raw90swap':
        return yaw + np.pi / 2.0, width, length
    if mode == 'negswap':
        return -yaw, width, length
    raise ValueError(f"unknown yaw mode {mode!r}; expected one of {YAW_MODES}")


class MmCows(VisionDataset):
    """VFA-facing MmCows dataset (single sequence).  See module docstring."""

    def __init__(self, root,  # path of ONE MmCows sequence directory
                 worldtrack_root=None,  # WorldTrack repo root (for `datasets`)
                 annotation_mode='3d',
                 heatmap_type='RGK',      # VFA 3D protocol uses rotated kernels
                 reload_heatmaps=False):  # True: ignore this sequence's .npy
        assert heatmap_type == 'RGK', \
            'MmCows adapter only builds RGK heatmaps (VFA 3D protocol)'
        super().__init__(root)
        self.__name__ = 'MmCows'
        self.root = root

        # ── wrap BEVine3D's MmCows3D (identical calibration & GT) ──────
        worldtrack_root = (worldtrack_root
                           or os.environ.get('WORLDTRACK_ROOT')
                           or '/home/mae/BEVine3D/WorldTrack')
        if not os.path.isdir(worldtrack_root):
            raise FileNotFoundError(
                f"WorldTrack repo root not found: {worldtrack_root!r}. "
                f"Pass worldtrack_root=... or set WORLDTRACK_ROOT.")
        if worldtrack_root not in sys.path:
            sys.path.insert(0, worldtrack_root)
        from datasets.mmcows3d_dataset import MmCows3D  # deferred: needs path
        self.base3d = MmCows3D(root, annotation_mode=annotation_mode)

        # ── VFA interface: geometry ────────────────────────────────────
        self.num_cam = 4
        # frame_list order preserved; frameDataset addresses frames by
        # POSITIONAL index 0..num_frame-1, so we keep the mapping.
        self.frame_list = list(self.base3d.frame_list)
        self.num_frame = len(self.frame_list)
        self.img_shape = [2800, 4480]           # native H, W
        self.image_size = (700, 1120)           # == resize_size (H, W)
        self.resize_size = (700, 1120)
        self.world_size = (1200, 1925)          # (length=y-ext, width=x-ext)
        self.cube_LWH = (25, 25, 32)
        self.reduced_grid_size = [48, 77]
        self.label_names = MMCOWS_BBOX_LABEL_NAMES

        # intrinsics scaled for the 0.25 resize; extrinsics untouched
        self.intrinsic_matrices = tuple(
            INTRINSIC_SCALE @ np.asarray(K, dtype=np.float64)
            for K in self.base3d.intrinsic_matrices)
        # WORLD_Z_SIGN: see module docstring.  -1 negates the z-column of
        # [R|t]; the z=0 floor projection is unchanged, while z in [0,+h]
        # then extends PHYSICALLY upward (cow bodies above the floor).
        ext = []
        for E in self.base3d.extrinsic_matrices:
            E = np.array(E, dtype=np.float64, copy=True)
            E[:, 2] *= WORLD_Z_SIGN
            ext.append(E)
        self.extrinsic_matrices = tuple(ext)

        # ── Stage 3: supervision targets (multiviewC.download() template) ──
        # Per-sequence AND per-grid-geometry RGK cache.  MultiviewC hardcodes
        # ONE fixed path (r'vfa/data/mc_RGK.npy'); a fixed path here would
        # collide across MmCows sequences, while a sequence-only path would
        # silently reuse heatmaps built with another world_size/cube_LWH.
        seq_name = os.path.basename(os.path.normpath(root))
        cache_name = rgk_cache_filename(seq_name, self.world_size,
                                        self.cube_LWH)
        self.RGK = RotationGaussianKernel(
            save_dir=os.path.join(MMCOWS_CACHE_DIR, cache_name))
        legacy_path = os.path.join(MMCOWS_CACHE_DIR,
                                   f'mmcows_RGK_{seq_name}.npy')
        if os.path.exists(legacy_path) and not self.RGK.RGKExist():
            print(f"  MmCows(VFA): ignoring legacy RGK cache {legacy_path!r}; "
                  f"geometry-qualified cache is {self.RGK.save_dir!r}.")
        self.reload_RGK = reload_heatmaps
        # ONE class-average file for the whole MmCows benchmark.  Built from
        # TRAIN sequences only (see download()); every split loads the file.
        self.classAverage = ClassAverage(
            classes=['Cow'],
            save_path=os.path.join(MMCOWS_CACHE_DIR, 'mmcows_ClAvg.json'))
        self.labels, self.heatmaps = self.download()

    # ------------------------------------------------------------------
    def get_image_fpaths(self, frame_range):
        """VFA convention: {cam 1..4: {positional_index: path}}.

        ``frame_range`` holds POSITIONAL indices (0..num_frame-1, cf.
        frameDataset.frame_range); the wrapped MmCows3D keys images by
        frame KEY (int or timestamp string), so translate via frame_list.
        """
        frame_range = list(frame_range)
        keys = [self.frame_list[i] for i in frame_range]
        base_fpaths = self.base3d.get_image_fpaths(set(keys))  # {cam0: {key: p}}
        img_fpaths = {cam: {} for cam in range(1, self.num_cam + 1)}
        n_missing = 0
        for cam0 in range(self.num_cam):
            for i, key in zip(frame_range, keys):
                p = base_fpaths.get(cam0, {}).get(key)
                if p is None:
                    n_missing += 1
                    continue
                img_fpaths[cam0 + 1][i] = p
        if n_missing:
            print(f"  MmCows(VFA): ⚠ {n_missing} (cam, frame) image(s) "
                  f"missing under {os.path.join(self.root, 'Image_subsets')}")
        return img_fpaths

    # ------------------------------------------------------------------
    def download(self):
        """Labels + per-frame RGK heatmaps, following multiviewC.download().

        Cows with no 3D box for a frame are simply absent from
        base3d.get_3d_boxes(frame_key) and therefore skipped.
        """
        # cls-avg file missing (True): aggregate it from the TRAIN sequences
        # of the split manifest; else load it — train/val/test adapters must
        # all see the same train-only mean.
        BuildClsAvg = not os.path.exists(self.classAverage.save_path)
        # per-sequence RGK cache missing (True): build heatmaps from THIS
        # sequence's 3D annotations; else load the cached array.
        BuildRGK = self.reload_RGK or not self.RGK.RGKExist()

        labels = []
        n_skip = 0
        for fk in self.frame_list:
            boxes = self.base3d.get_3d_boxes(fk)  # {cow_id(int): box dict}
            cow_infos = []
            rgk_heatmap = np.zeros(self.reduced_grid_size, dtype=np.float32)
            for cid in sorted(boxes):
                obj = self._box_to_obj(boxes[cid], fk, cid)
                if obj is None:
                    n_skip += 1
                    continue
                cow_infos.append(obj)
                if BuildRGK:
                    rgk_heatmap = self._stamp_cow_kernel(rgk_heatmap, obj)
            if BuildRGK:
                self.RGK.add_item(rgk_heatmap)
            labels.append(cow_infos)
        if n_skip:
            print(f"  MmCows(VFA): skipped {n_skip} malformed 3D box(es) "
                  f"in total.")

        if BuildClsAvg:
            self._build_class_average_from_train_sequences()
        else:
            self.classAverage.load_from_file()

        if BuildRGK:
            heatmaps = self.RGK.dump_to_file()  # (num_frame, 48, 77)
        else:
            heatmaps = self.RGK.load_from_file()
        expected_shape = (self.num_frame, *self.reduced_grid_size)
        if not isinstance(heatmaps, np.ndarray) \
                or heatmaps.shape != expected_shape:
            got = (None if heatmaps is None else
                   getattr(heatmaps, 'shape', type(heatmaps).__name__))
            raise RuntimeError(
                f"MmCows RGK cache shape mismatch for {self.RGK.save_dir}: "
                f"expected {expected_shape} from world_size={self.world_size} "
                f"and cube_LWH={self.cube_LWH}, got {got}. Delete that cache "
                f"file and rerun so the targets are rebuilt for the current "
                f"grid.")
        return labels, heatmaps

    # ------------------------------------------------------------------
    @staticmethod
    def _box_to_obj(b, frame_key, cid):
        """One 3D-annotation box dict -> VFA Obj3D (VFA frame, [h,w,l])."""
        try:
            cx = float(b['center'][0])
            cy = float(b['center'][1])
            length = float(b['length'])
            width = float(b['width'])
            height = float(b['height'])
            yaw = float(b['yaw'])
        except (KeyError, TypeError, ValueError, IndexError) as e:
            print(f"  MmCows(VFA): ⚠ frame {frame_key} cow {cid}: "
                  f"malformed 3D box ({e}) — skipped")
            return None
        rotation, length, width = apply_yaw_convention(
            yaw, length, width, YAW_MODE)
        return Obj3D(
            classname='Cow',
            location=(cx - WORLD_X_MIN_CM,   # = cx + 879
                      cy - WORLD_Y_MIN_CM,   # = cy + 646
                      0.0),
            dimension=[height, width, length],   # VFA: [h, w, l]
            rotation=rotation,
            conf=None)

    # ------------------------------------------------------------------
    def _stamp_cow_kernel(self, heatmap, obj):
        """RGK.gaussian_kernel_heatmap for one cow (25-cm cell units).

        box_cx/box_cy are the cow centre in heatmap CELLS (25 cm each):
        the adapter stores location = (x_cm + 879, y_cm + 646) and
        1925/25 = 77 cells (x), 1200/25 = 48 cells (y), so dividing the
        stored location by cube_LWH[:2] is EXACTLY VFA's _assign_to_grid
        normalization (location / world_size * grid_size) — the same
        cell-CORNER convention MultiviewC uses.  l, w are passed in 25-cm
        CELLS (length/25, width/25); yaw is passed in DEGREES, as
        RGK.bi_rotate expects.
        """
        x, y = obj.location[:2]
        _, w, l = obj.dimension                        # [h, w, l] cm
        box_cx = x / self.cube_LWH[0]
        box_cy = y / self.cube_LWH[1]
        l_cells = l / self.cube_LWH[0]
        w_cells = w / self.cube_LWH[1]
        yaw_deg = float(np.rad2deg(obj.rotation))
        return self.RGK.gaussian_kernel_heatmap(
            heatmap, box_cx, box_cy, l_cells, w_cells, yaw_deg)

    # ------------------------------------------------------------------
    def _train_sequence_dirs(self):
        """Sequence dirs of the 'train' split per the split manifest.

        Used ONLY to build the class-average file; val/test sequences must
        not contribute to the mean.  Falls back to this sequence alone (with
        a loud warning) if the manifest is unreadable.
        """
        data_root = os.path.dirname(os.path.normpath(self.root))
        manifest_path = os.path.join(data_root, SPLIT_MANIFEST_NAME)
        if not os.path.exists(manifest_path):
            print(f"  MmCows(VFA): ⚠ split manifest not found at "
                  f"{manifest_path!r} — building classAverage from THIS "
                  f"sequence only. Rebuild later: delete "
                  f"{self.classAverage.save_path} and re-run on a train "
                  f"sequence.")
            return [os.path.normpath(self.root)]
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        if 'train' not in manifest:
            raise KeyError(f"'train' not in {manifest_path}; available "
                           f"splits: {list(manifest.keys())}")
        dirs = [os.path.join(data_root, str(name))
                for name in manifest['train']]
        dirs = [d for d in dirs if os.path.isdir(d)]
        if not dirs:
            print(f"  MmCows(VFA): ⚠ no train sequence dir from "
                  f"{manifest_path} exists under {data_root!r} — falling "
                  f"back to THIS sequence only.")
            return [os.path.normpath(self.root)]
        return dirs

    # ------------------------------------------------------------------
    def _build_class_average_from_train_sequences(self):
        """Aggregate ClassAverage over TRAIN sequences, then dump to file.

        Dimension order is VFA's [h, w, l] in cm — identical to what
        _encode_dimension / decode3d consume via Obj3D.dimension.
        """
        Base3D = type(self.base3d)          # the wrapped MmCows3D class
        dirs = self._train_sequence_dirs()
        print(f"  MmCows(VFA): building classAverage over {len(dirs)} "
              f"TRAIN sequence(s): {[os.path.basename(d) for d in dirs]}")
        for d in dirs:
            tmp = Base3D(d, annotation_mode='3d')
            for fk in tmp.frame_list:
                boxes = tmp.get_3d_boxes(fk)
                for cid in sorted(boxes):
                    b = boxes[cid]
                    try:
                        dim = [float(b['height']), float(b['width']),
                               float(b['length'])]               # [h, w, l]
                    except (KeyError, TypeError, ValueError):
                        continue
                    self.classAverage.add_item('Cow', dim)
        self.classAverage.dump_to_file()
        mean = self.classAverage.get_mean('Cow')
        print(f"  MmCows(VFA): classAverage mean [h,w,l] = "
              f"{np.round(mean, 2).tolist()} cm from "
              f"{self.classAverage.dimension_map['cow']['count']} boxes; "
              f"saved -> {self.classAverage.save_path}")
