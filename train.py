import os, random
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
import matplotlib
matplotlib.use('agg')
from tensorboardX import SummaryWriter
from torchvision import transforms
import shutil

from vfa.model.vfanet import VFANet
from vfa.model.vfa_op import mmcows_convert
from vfa.trainer import Trainer
from torch.utils.data import DataLoader, Subset
from vfa.utils import collate, to_numpy
from vfa.data.dataset import frameDataset, MultiviewC, MultiviewX, Wildtrack
from vfa.data.encoder import ObjectEncoder
from vfa.data.multiseq_dataset import build_multiseq
from vfa.data.mmcows import MMCOWS_CACHE_DIR, rgk_cache_filename
from vfa.config import *

def parse(opts):

    parser = ArgumentParser()

    #Data options
    parser.add_argument('--root', type=str, default=opts.root,
                        help='root directory of dataset')

    parser.add_argument('--data', type=str, default=opts.name,
                        help='the name of dataset')                        

    parser.add_argument('--mode', type=str, default=opts.mode,
                        help='2D/3D mode determines the detection task.')
    
    # MultiviewC: (3900, 3900), MultiviewX: (640, 1000)
    parser.add_argument('--world_size', type=int, nargs=2, default=opts.world_size, 
                        help='width and length of designed grid')

    # MultiviewC: (720, 1080), MultiviewX: (1080, 1920)
    parser.add_argument('--image_size', type=int, nargs=2, default=opts.image_size,
                        help='height and width of image')
    
    parser.add_argument('--resize_size', type=int, nargs=2, default=opts.resize_size,
                        help='resized height and width of image')

    parser.add_argument('--ann', type=str, default=getattr(opts, 'ann', 'annotations_positions'),
                        help='annotation of MultiviewC dataset')

    parser.add_argument('--calib', type=str, default=getattr(opts, 'calib', 'calibrations'),
                        help='calibrations of MultiviewC dataset')

    # Training options
    parser.add_argument('-e', '--epochs', type=int, default=40,
                        help='the number of epochs for training')
    
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='batch size for training. [NOTICE]: this repo only support \
                              batch size of 1')

    parser.add_argument('--lr', type=float, default=0.02,
                        help='learning rate')
    
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='learning rate')
    
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum')
    

    # Model options
    # MultiviewC: 160, MultiviewX: 32
    parser.add_argument('--grid_h', type=int, default=opts.grid_h, 
                    help='height of designed grid')

    # MultiviewC: (25, 25, 32), MultiviewX: (4, 4, 4)
    parser.add_argument('--cube_size', type=int, default=opts.cube_size,  
                        help='the size of cube of designed grid')

    parser.add_argument('--grid_scale', type=int, default=opts.grid_scale,
                        help='make the ratio and scale of grid correspond, \
                              which also project the design voxel to image successfully.')

    parser.add_argument('--topdown', type=int, default=0, # discarded
                        help='the number of residual blocks in topdown network')

    parser.add_argument('--angle_range', type=int, default=360,
                        help='the range of angle prediction for circle smooth label (CSL)')

    parser.add_argument('--pretrained', type=bool, default=True,
                        help='load the pretrained checkpoint of feature extractor eg. resnet18')  
                          
    parser.add_argument('--heatmap', type=str, default='GK',
                        help='the type of heatmap, `RGK`, rotated gaussian kernel heatmap,\
                              or `GK`, normal gaussian kernel')       

    # Training options
    parser.add_argument('--seed', type=int, default=1, 
                        help='random seed')

    parser.add_argument('--savedir', type=str,
                        default='experiments')
    
    parser.add_argument('--resume', type=str,
                        default=None)
    
    parser.add_argument('--checkpoint', type=str,
                        default=None)

    parser.add_argument('--overfit_frames', type=int, default=0,
                        help='MmCows smoke test: train/validate on the first N '
                             'frames of ONE train sequence, without shuffling. '
                             '0 disables the smoke test.')
    # Experiment options
    # MultiviewC 3D detection: heatmap, location, dimension and rotation. loss_weight has 4 weights in total.
    # MultiviewX 2D detection: heatmap and location. loss_weight has 2 weights in total.
    parser.add_argument('--loss_weight', type=float, nargs=4, default=opts.loss_weight,
                        help='the 3D weight of each loss including heatmap, location, dimension and rotation;\
                             or 2D weight of each loss only including heatmap and location.')

    parser.add_argument('--print_iter', type=int, default=1,
                        help='print loss summary every N iterations')

    parser.add_argument('--vis_iter', type=int, default=50,
                        help='display visualizations every N iterations')

    parser.add_argument('--cls_thresh', type=float, default=0.8,
                        help='positive sample confidence threshold')                        

    parser.add_argument('--topk', type=int, default=50,
                        help='the number of positive samples after nms')                        
    
    parser.add_argument('--start_save', type=int, default=5,
                        help='After `start_save` epochs, model starts to save.')
    
    parser.add_argument('--copy_repo', type=bool, default=True,
                        help='Copy the whole repo before training')

    args = parser.parse_args()
    print('Settings:')
    print(vars(args))
    return args

def setup_seed(seed=7777):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

def make_experiment(args, copy_repo=False):
    lastdir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # lastdir = 'TestRepo'
    args.savedir = os.path.join(args.savedir , lastdir)
    summary = SummaryWriter(args.savedir+'/tensorboard')
    summary.add_text('config', '\n'.join(
        '{:12s} {}'.format(k, v) for k, v in sorted(args.__dict__.items())))
    summary.file_writer.flush()
    if copy_repo:
        os.makedirs(args.savedir, exist_ok=True)
        shutil.copytree('./vfa', args.savedir + '/scripts/mfot3d', dirs_exist_ok=True)
    return summary, args

def resume_experiment(args):
    summary_dir = os.path.join(args.savedir, args.resume, 'tensorboard')
    args.savedir = os.path.join(args.savedir, args.resume)
    summary = SummaryWriter(summary_dir)
    return summary, args

def save(model, epoch, args, optimizer, scheduler, train_loss, val_loss):
    savedir = os.path.join(args.savedir, 'checkpoints')
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    checkpoints = {
        'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'scheduler_state_dict' : scheduler.state_dict(),
        'args':args
    }
    torch.save(checkpoints, os.path.join(savedir, 'Epoch{:02d}_train_loss{:.4f}_val_loss{:.4f}.pth'.\
                        format(epoch, train_loss['loss'], val_loss['loss'])))

def resume(resume_dir, model, optimizer, scheduler, device):
    checkpoints = torch.load(resume_dir)
    pretrain = checkpoints['model_state_dict']
    current = model.state_dict()
    state_dict = {k: v for k, v in pretrain.items() if k in current.keys()}
    current.update(state_dict)
    model.load_state_dict(current)

    optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
    epoch = checkpoints['epoch'] + 1
    print("Model resume training from %s" %resume_dir)
    return model, optimizer, scheduler, epoch

def _unwrap_multiseq(dataset):
    """Return the underlying multi-sequence dataset from a Subset."""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def _select_one_sequence_indices(dataset, n_frames):
    """Deterministically choose N frames from the first sequence with N frames."""
    if n_frames <= 0:
        raise ValueError(f'--overfit_frames must be positive, got {n_frames}')
    base = _unwrap_multiseq(dataset)
    if hasattr(base, 'sequence_boundaries'):
        names = getattr(base, 'sequence_names',
                        [f'sequence_{i}' for i in range(len(base.sequence_boundaries))])
        for seq_name, (start, end) in zip(names, base.sequence_boundaries):
            if end - start >= n_frames:
                return list(range(start, start + n_frames)), seq_name
        available = ', '.join(f'{name}: {end - start}'
                              for name, (start, end) in zip(names, base.sequence_boundaries))
        raise ValueError(f'No train sequence has {n_frames} frames ({available})')
    if len(base) < n_frames:
        raise ValueError(f'Dataset has only {len(base)} frames, cannot overfit {n_frames}')
    return list(range(n_frames)), getattr(base, 'root', 'single_sequence')


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _object_xy(objects):
    xy = []
    for obj in objects:
        loc = np.asarray(to_numpy(obj.location), dtype=np.float64).reshape(-1)
        xy.append(loc[:2])
    return (np.stack(xy, axis=0) if xy else np.zeros((0, 2), dtype=np.float64))


def _match_predictions(gt_objects, pred_objects, max_dist_cm=25.0):
    """One-to-one GT/prediction matching in the VFA metric grid frame."""
    gt_xy = _object_xy(gt_objects)
    pred_xy = _object_xy(pred_objects)
    if len(gt_xy) == 0:
        return 0, []
    if len(pred_xy) == 0:
        return 0, []
    from scipy.optimize import linear_sum_assignment
    distances = np.linalg.norm(gt_xy[:, None, :] - pred_xy[None, :, :], axis=2)
    gt_idx, pred_idx = linear_sum_assignment(distances)
    matches = [(int(g), int(p), float(distances[g, p]))
               for g, p in zip(gt_idx, pred_idx)
               if distances[g, p] <= max_dist_cm]
    return len(matches), matches


def _save_heatmap_overlay(path, gt_heatmap, pred_heatmap, gt_objects,
                          pred_objects, cube_size, title):
    """Save GT and predicted heatmaps with GT/predicted centres overlaid."""
    gt_map = np.asarray(to_numpy(gt_heatmap), dtype=np.float32)
    pred_map = np.asarray(to_numpy(pred_heatmap), dtype=np.float32)
    gt_xy = _object_xy(gt_objects)
    pred_xy = _object_xy(pred_objects)
    cell_x, cell_y = float(cube_size[0]), float(cube_size[1])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, heatmap, name in zip(axes, (gt_map, pred_map), ('GT RGK', 'Prediction')):
        im = ax.imshow(heatmap, cmap='viridis', vmin=0.0, vmax=1.0,
                       origin='upper', interpolation='nearest')
        ax.set_title(name)
        ax.set_xlabel('x cell')
        ax.set_ylabel('y cell')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if len(gt_xy):
        axes[0].scatter(gt_xy[:, 0] / cell_x, gt_xy[:, 1] / cell_y,
                        s=45, facecolors='none', edgecolors='lime', linewidths=1.5,
                        label='GT')
        axes[1].scatter(gt_xy[:, 0] / cell_x, gt_xy[:, 1] / cell_y,
                        s=45, facecolors='none', edgecolors='lime', linewidths=1.5,
                        label='GT')
    if len(pred_xy):
        axes[1].scatter(pred_xy[:, 0] / cell_x, pred_xy[:, 1] / cell_y,
                        s=45, marker='x', c='red', linewidths=1.5,
                        label='pred')
    axes[1].legend(loc='upper right')
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_heatmap_image(path, heatmap, title):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    im = ax.imshow(np.asarray(to_numpy(heatmap), dtype=np.float32),
                   cmap='viridis', vmin=0.0, vmax=1.0,
                   origin='upper', interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('x cell')
    ax.set_ylabel('y cell')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_grid_projection(path, image, calib, grid, objects):
    """Project this run's training grid and GT centres onto camera 1."""
    image_np = np.asarray(to_numpy(image), dtype=np.float32)
    image_np = np.clip(image_np.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(image_np)
    draw = ImageDraw.Draw(canvas)

    grid_pts = np.asarray(to_numpy(grid), dtype=np.float64).reshape(-1, 3).copy()
    grid_pts = mmcows_convert(grid_pts, scale=1.)
    calib_np = np.asarray(to_numpy(calib), dtype=np.float64)
    homogeneous = np.concatenate([grid_pts, np.ones((len(grid_pts), 1))], axis=1)
    projected = homogeneous @ calib_np.T
    depth = projected[:, 2]
    uv = projected[:, :2] / np.maximum(depth[:, None], 1e-6)
    h, w = image_np.shape[:2]

    for (u, v), d in zip(uv[::4], depth[::4]):
        if d > 1.0 and 0 <= u < w and 0 <= v < h:
            draw.ellipse([u - 1, v - 1, u + 1, v + 1], fill=(0, 255, 255))

    rows, cols = grid.shape[:2]
    boundary = [(0, 0), (0, cols - 1), (rows - 1, cols - 1),
                (rows - 1, 0), (0, 0)]
    for (r0, c0), (r1, c1) in zip(boundary[:-1], boundary[1:]):
        i0, i1 = r0 * cols + c0, r1 * cols + c1
        if depth[i0] > 1.0 and depth[i1] > 1.0:
            draw.line([tuple(uv[i0]), tuple(uv[i1])], fill=(255, 255, 0), width=2)

    gt_xy = _object_xy(objects)
    if len(gt_xy):
        gt_pts = np.concatenate([
            gt_xy - np.array([[879.0, 646.0]]),
            np.zeros((len(gt_xy), 1)),
            np.ones((len(gt_xy), 1))], axis=1)
        gt_proj = gt_pts @ calib_np.T
        gt_depth = gt_proj[:, 2]
        gt_uv = gt_proj[:, :2] / np.maximum(gt_depth[:, None], 1e-6)
        for (u, v), d in zip(gt_uv, gt_depth):
            if d > 1.0 and 0 <= u < w and 0 <= v < h:
                draw.ellipse([u - 4, v - 4, u + 4, v + 4],
                             outline=(255, 0, 0), width=2)
    canvas.save(path)


def _run_overfit_evaluation(model, encoder, dataset, args, device, n_vis=4):
    """Decode the final overfit model and enforce the 25-cm recovery gate."""
    vis_dir = os.path.join(args.savedir, 'overfit_vis')
    os.makedirs(vis_dir, exist_ok=True)
    cls_thresh = 0.5
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=collate)
    positions = np.linspace(0, len(dataset) - 1, num=min(n_vis, len(dataset)))
    positions = sorted(set(int(round(p)) for p in positions))
    base = _unwrap_multiseq(dataset)
    reports, saved = [], []

    model.eval()
    with torch.no_grad():
        for position, (_, images, objects, heatmaps, calibs, grid) in enumerate(loader):
            if position not in positions:
                continue
            images = images.to(device)
            calibs = calibs.to(device)
            grid = grid.to(device)
            encoded_pred = model(images, calibs, grid)
            pred_objects = encoder.batch_decode(encoded_pred, cls_thresh)
            gt_objects = objects[0]
            matched, matches = _match_predictions(gt_objects, pred_objects,
                                                  max_dist_cm=25.0)
            global_idx = (dataset.indices[position]
                          if isinstance(dataset, Subset) else position)
            seq_name = 'sequence'
            if hasattr(base, 'sequence_names'):
                for name, (start, end) in zip(base.sequence_names,
                                              base.sequence_boundaries):
                    if start <= global_idx < end:
                        seq_name = name
                        break
            out_path = os.path.join(
                vis_dir,
                f'{seq_name}_subset{position:03d}_global{global_idx:04d}_heatmap.png')
            _save_heatmap_overlay(
                out_path, heatmaps[0], torch.sigmoid(encoded_pred['heatmap'])[0, 0],
                gt_objects, pred_objects, args.cube_size,
                f'{seq_name} subset {position} | GT {len(gt_objects)} | '
                f'pred {len(pred_objects)} | matched {matched}')
            report = {
                'subset_position': int(position),
                'global_index': int(global_idx),
                'sequence': seq_name,
                'gt': len(gt_objects),
                'predictions': len(pred_objects),
                'matched_within_25cm': matched,
                'matches_gt_pred_distance_cm': matches,
                'overlay': out_path,
            }
            reports.append(report)
            saved.append(out_path)
            print(f"  [OVERFIT VIS] {seq_name} subset {position}: "
                  f"{matched}/{len(gt_objects)} GT cows recovered within 25 cm "
                  f"({len(pred_objects)} predictions) -> {out_path}")
            if len(reports) == len(positions):
                break

    total_gt = sum(r['gt'] for r in reports)
    total_matched = sum(r['matched_within_25cm'] for r in reports)
    recovery = total_matched / max(total_gt, 1)
    result = {
        'cls_thresh': cls_thresh,
        'frames_evaluated': len(reports),
        'gt_cows': total_gt,
        'matched_within_25cm': total_matched,
        'recovery': recovery,
        'pass_recovery': recovery >= 0.90,
        'frames': reports,
        'overlays': saved,
    }
    summary_path = os.path.join(vis_dir, 'overfit_eval.json')
    with open(summary_path, 'w') as f:
        json.dump(_json_safe(result), f, indent=2)
    print(f"  [OVERFIT VIS] total: {total_matched}/{total_gt} GT cows recovered "
          f"within 25 cm ({100.0 * recovery:.2f}%), cls_thresh={cls_thresh}; "
          f"summary -> {summary_path}")
    return result


def _dump_overfit_diagnostics(model, encoder, dataset, args, device,
                              first_train_loss, final_train_loss,
                              final_val_loss, eval_result):
    """Dump geometry/target diagnostics before any further code changes."""
    diag_dir = os.path.join(args.savedir, 'overfit_diagnostics')
    os.makedirs(diag_dir, exist_ok=True)
    print(f"\n[OVERFIT DIAGNOSTICS] writing failure diagnostics to {diag_dir}")

    loss_report = {
        'first_train_loss_terms': first_train_loss,
        'final_train_loss_terms': final_train_loss,
        'final_val_loss_terms': final_val_loss,
        'loss_reduction_required': '>= 10x',
        'actual_train_loss_reduction': (
            first_train_loss['loss'] / max(final_train_loss['loss'], 1e-12)),
        'eval_result': eval_result,
    }
    loss_path = os.path.join(diag_dir, 'loss_terms.json')
    with open(loss_path, 'w') as f:
        json.dump(_json_safe(loss_report), f, indent=2)
    print('  per-loss terms:')
    print(json.dumps(_json_safe(loss_report), indent=2))

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=collate)
    _, images, objects, heatmaps, calibs, grid = next(iter(loader))
    model.eval()
    with torch.no_grad():
        encoded_pred = model(images.to(device), calibs.to(device), grid.to(device))
    gt_path = os.path.join(diag_dir, 'frame000_gt_heatmap.png')
    pred_path = os.path.join(diag_dir, 'frame000_pred_heatmap.png')
    grid_path = os.path.join(diag_dir, 'frame000_cam1_grid_projection.png')
    _save_heatmap_image(gt_path, heatmaps[0], 'GT RGK heatmap')
    _save_heatmap_image(pred_path, torch.sigmoid(encoded_pred['heatmap'])[0, 0],
                        'Predicted sigmoid heatmap')
    _save_grid_projection(grid_path, images[0], calibs[0], grid[0], objects[0])

    cache_report = []
    base = _unwrap_multiseq(dataset)
    frame_datasets = getattr(base, 'datasets', [base])
    for frame_ds in frame_datasets:
        adapter = getattr(frame_ds, 'base', None)
        if adapter is None or adapter.__name__ != 'MmCows':
            continue
        seq_name = os.path.basename(os.path.normpath(adapter.root))
        expected_name = rgk_cache_filename(seq_name, adapter.world_size,
                                           adapter.cube_LWH)
        expected_path = os.path.join(MMCOWS_CACHE_DIR, expected_name)
        legacy_path = os.path.join(MMCOWS_CACHE_DIR,
                                   f'mmcows_RGK_{seq_name}.npy')
        cache_report.append({
            'sequence': seq_name,
            'world_size': adapter.world_size,
            'cube_LWH': adapter.cube_LWH,
            'reduced_grid_size': adapter.reduced_grid_size,
            'expected_cache': expected_path,
            'actual_cache': adapter.RGK.save_dir,
            'cache_key_is_geometry_qualified': os.path.basename(adapter.RGK.save_dir) == expected_name,
            'actual_cache_exists': os.path.exists(adapter.RGK.save_dir),
            'legacy_sequence_only_cache_exists': os.path.exists(legacy_path),
            'heatmap_shape': np.shape(adapter.heatmaps),
            'expected_heatmap_shape': (adapter.num_frame, *adapter.reduced_grid_size),
        })
    cache_path = os.path.join(diag_dir, 'rgk_cache_manifest.json')
    with open(cache_path, 'w') as f:
        json.dump(_json_safe(cache_report), f, indent=2)
    print(f"  saved: {loss_path}")
    print(f"  saved: {gt_path}")
    print(f"  saved: {pred_path}")
    print(f"  saved: {grid_path}")
    print(f"  saved: {cache_path}")

def train(opts):
    # Parse commond argument
    args = parse(opts)
    
    # Setup random seed
    setup_seed(args.seed)

    #TODO: Add view-coherent data augmentation
    # Data augmentaion for training dataset 
    train_transform = transforms.Compose([transforms.Resize(args.resize_size),
                                          transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
                                          transforms.ToTensor()])

    val_transform = transforms.Compose([transforms.Resize(args.resize_size),
                                          transforms.ToTensor()])

    # Create datasets
    if opts.name == 'MultiviewC':
        train_data = frameDataset(MultiviewC(root=args.root, heatmap_type=args.heatmap, 
                                             ann_root=args.ann, calib_root=args.calib, 
                                             world_size=args.world_size, cube_LWH=args.cube_size), 
                                             transform=train_transform, split='train')
        
        val_data = frameDataset(MultiviewC(root=args.root, heatmap_type=args.heatmap, 
                                           ann_root=args.ann, calib_root=args.calib, 
                                           world_size=args.world_size, cube_LWH=args.cube_size),
                                           transform=val_transform, split='val')
    elif opts.name == 'MultiviewX':
        train_data = frameDataset(MultiviewX(root=args.root, world_size=args.world_size, cube_LWH=args.cube_size), 
                                             transform=train_transform, split='train')
        
        val_data = frameDataset(MultiviewX(root=args.root, world_size=args.world_size, cube_LWH=args.cube_size),
                                           transform=val_transform, split='val')
    
    elif opts.name == 'Wildtrack':
        train_data = frameDataset(Wildtrack(root=args.root, world_size=args.world_size, cube_LWH=args.cube_size), 
                                             transform=train_transform, split='train')
        
        val_data = frameDataset(Wildtrack(root=args.root, world_size=args.world_size, cube_LWH=args.cube_size),
                                           transform=val_transform, split='val')
    elif opts.name == 'MmCows':
        overfit = args.overfit_frames > 0
        # Full MmCows training follows BEVine3D's stationary-frame protocol.
        # The overfit smoke test deliberately disables it so "the first N
        # frames of one sequence" is fixed and reproducible.
        train_data = build_multiseq(args.root,
                                    manifest_name=opts.manifest_name,
                                    split='train',
                                    transform=train_transform,
                                    drop_stationary=not overfit)
        if overfit:
            # Same train frames for validation, but through val_transform
            # (no ColorJitter), so the validation number is comparable.
            val_data = build_multiseq(args.root,
                                      manifest_name=opts.manifest_name,
                                      split='train',
                                      transform=val_transform,
                                      drop_stationary=False)
        else:
            val_data = build_multiseq(args.root,
                                      manifest_name=opts.manifest_name,
                                      split='val',
                                      transform=val_transform,
                                      drop_stationary=False)

    if args.overfit_frames:
        overfit_indices, overfit_sequence = _select_one_sequence_indices(
            train_data, args.overfit_frames)
        train_data = Subset(train_data, overfit_indices)
        val_data = Subset(val_data, overfit_indices)
        print(f"  [OVERFIT] using {len(overfit_indices)} fixed frames from "
              f"ONE train sequence '{overfit_sequence}' "
              f"(global indices {overfit_indices[0]}..{overfit_indices[-1]}); "
              f"the same subset is used for validation and shuffling is OFF.")

    # Create dataloader
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=args.overfit_frames == 0,
                              num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    # Device: default 1 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build model
    model = VFANet(args=args, grid_height=args.grid_h, cube_size=args.cube_size, angle_range=args.angle_range,
                    mode=args.mode, pretrained=args.pretrained).to(device)

    # Create encoder
    encoder = ObjectEncoder(train_data, topk=args.topk)

    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), 
                                              epochs=args.epochs)

    # Create Summary & Resume Training
    if args.resume is not None:
        summary, args = resume_experiment(args)
        resume_dir = os.path.join(args.savedir, 'checkpoints', args.checkpoint)
        model, optimizer, scheduler, start = \
            resume(resume_dir, model, optimizer, scheduler, device)
    else:
        summary, args = make_experiment(args, args.copy_repo)
        start = 1

    # Create Trainer
    trainer = Trainer(model, args, device, summary, args.loss_weight)

    first_train_loss = None
    final_train_loss = None
    final_val_loss = None
    for epoch in range(start, args.epochs + 1):
        scheduler.step()
        summary.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Train model
        train_loss = trainer.train(train_loader, encoder, optimizer, epoch, args)
        if first_train_loss is None:
            first_train_loss = dict(train_loss)
        final_train_loss = dict(train_loss)

        # Validate model
        val_loss = trainer.validate(val_loader, encoder, epoch, args)
        final_val_loss = dict(val_loss)

        summary.add_scalars('loss', {'train_loss': train_loss['loss'], 'val_loss': val_loss['loss']}, epoch)
        # if epoch > args.start_save:
        if epoch % 5 == 0:
            save(model, epoch, args, optimizer, scheduler, train_loss, val_loss)

    if args.overfit_frames:
        print('\n=== MmCows overfit smoke-test gates ===')
        eval_result = _run_overfit_evaluation(model, encoder, val_data, args,
                                              device, n_vis=4)
        loss_ratio = (first_train_loss['loss'] /
                      max(final_train_loss['loss'], 1e-12))
        loss_pass = final_train_loss['loss'] <= first_train_loss['loss'] / 10.0
        recovery_pass = eval_result['pass_recovery']
        print(f"  loss gate: epoch {start} total {first_train_loss['loss']:.6f} "
              f"-> epoch {args.epochs} total {final_train_loss['loss']:.6f}; "
              f"reduction {loss_ratio:.2f}x (required >= 10x): "
              f"{'PASS' if loss_pass else 'FAIL'}")
        print(f"  decode gate: {eval_result['matched_within_25cm']}/"
              f"{eval_result['gt_cows']} GT cows recovered within 25 cm "
              f"({100.0 * eval_result['recovery']:.2f}%; required >= 90%): "
              f"{'PASS' if recovery_pass else 'FAIL'}")
        if not (loss_pass and recovery_pass):
            summary.file_writer.flush()
            summary.close()
            _dump_overfit_diagnostics(model, encoder, val_data, args, device,
                                      first_train_loss, final_train_loss,
                                      final_val_loss, eval_result)
            raise SystemExit('MmCows overfit smoke test FAILED; diagnostics were '
                             'saved before any further changes.')
        print('MmCows overfit smoke test PASSED.')

    summary.file_writer.flush()
    summary.close()


if __name__ == '__main__':
    mode_parser = ArgumentParser()
    mode_parser.add_argument('--data', type=str, required=True, 
                        help='dataset: MultiviewC, MultiviewX, Wildtrack, MmCows')
    # parse_args() would reject training flags such as --overfit_frames and -e
    # before train()'s full parser can see them.
    mode, _ = mode_parser.parse_known_args()
    if mode.data == mc_opts.name:
        # MultiviewC
        train(mc_opts)
    elif mode.data == mx_opts.name:
        # MultiviewX
        train(mx_opts)
    elif mode.data == wt_opts.name:
        # Wildtrack
        train(wt_opts)
    elif mode.data == mmcows_opts.name:
        # MmCows
        train(mmcows_opts)
    else:
        raise ValueError('Dataset error, expect `MultiviewC`, `MultiviewX`, `Wildtrack`, `MmCows`, got {}.'.format(mode.data))
    

   
        
