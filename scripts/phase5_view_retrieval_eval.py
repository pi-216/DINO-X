#!/usr/bin/env python3
"""Phase 5: Label-free view-retrieval evaluation.

Goal: objectively validate CT representations without labels.

Protocol:
- Sample N rows from the eval split (val.series_dir from split manifest).
- For each row, generate two views using Phase 5's PngDataset (same augmentations).
- Embed both views with the backbone (CLS token, pre-projection head).
- Compute retrieval accuracy: for query i (view1), does nearest key in view2 match i?

Options:
- --vit-layer N: extract CLS from intermediate layer N (arXiv:2604.23670)
- --pos-bias-project: project out positional bias via PCA on noise (arXiv:2604.23670)

Outputs:
- Prints ok=true, passed=true|false and metrics.
- Writes a JSON metrics file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _need(mod: str) -> None:
    raise SystemExit(f"Missing dependency: {mod}. Install it (e.g., into .venv) and retry.")


try:
    import numpy as np
except Exception:
    _need("numpy")

try:
    import torch
    import torch.nn.functional as F
except Exception:
    _need("torch")


def _load_phase5_module() -> Any:
    p = Path(__file__).resolve().parent / "phase5_big_run.py"
    spec = importlib.util.spec_from_file_location("phase5_big_run", p)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Failed to load module spec from: {p}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _load_split_series_dirs(split_manifest: Path, which: str) -> set[str]:
    payload = json.loads(split_manifest.read_text())
    series = payload.get(which, {}).get("series_dir", [])
    if not isinstance(series, list) or not series:
        raise SystemExit(f"Invalid split manifest (missing {which}.series_dir): {split_manifest}")
    return set(str(s) for s in series)


@torch.no_grad()
def _embed_backbone_cls(student: Any, x: torch.Tensor, spacing: torch.Tensor | None = None, vit_layer: int | None = None) -> torch.Tensor:
    """Return L2-normalized CLS embedding from backbone.

    If vit_layer is specified (0-indexed), extracts the CLS token after
    that transformer block instead of the final backbone output. Layer 0
    means after the first block. Per arXiv:2604.23670, middle layers
    preserve spatial correspondence better than the final layer.
    """
    if vit_layer is None:
        feats = student.backbone(x, spacing=spacing)  # (B, N+1, D)
        cls = feats[:, 0]
        return F.normalize(cls.float(), p=2, dim=-1)

    # Manual forward through backbone up to layer `vit_layer`
    vit = student.backbone
    B = x.size(0)
    y = vit.patch_embed(x)
    y = y.flatten(2).transpose(1, 2)
    cls_tokens = vit.cls_token.expand(B, -1, -1)
    y = torch.cat([cls_tokens, y], dim=1)
    y = y + vit.pos_embed
    if vit.scale_aware and spacing is not None:
        y = y + vit.scale_embed(spacing)
    for i, blk in enumerate(vit.blocks):
        y = blk(y)
        if i == vit_layer:
            break
    cls = y[:, 0]
    return F.normalize(cls.float(), p=2, dim=-1)


@torch.no_grad()
def _compute_pos_bias_dir(student: Any, device: torch.device, vit_layer: int, dim: int, img_size: int = 224, n_dirs: int = 1) -> np.ndarray:
    """Compute positional bias direction via PCA on noise-image activations.

    Per arXiv:2604.23670, DINO features contain a stable positional artifact
    that interferes with geometric matching. This function computes the
    dominant eigenvector of patch features from a noise image, which
    captures the positional bias.

    Returns a matrix of shape (dim, n_dirs) — the top n_dirs eigenvectors.
    """
    noise = torch.randn(1, 3, img_size, img_size, device=device)

    vit = student.backbone
    B = noise.size(0)
    y = vit.patch_embed(noise)
    y = y.flatten(2).transpose(1, 2)
    cls_tokens = vit.cls_token.expand(B, -1, -1)
    y = torch.cat([cls_tokens, y], dim=1)
    y = y + vit.pos_embed
    # Forward through blocks up to vit_layer
    for i, blk in enumerate(vit.blocks):
        y = blk(y)
        if i == vit_layer:
            break
    # Use CLS token (shape: (1, dim))
    cls_feat = y[:, 0]  # (1, dim)
    # Center
    cls_feat_centered = cls_feat - cls_feat.mean(dim=0, keepdim=True)
    # SVD on single sample — equivalent to using it as the direction
    # For more robust estimate, we use the feature itself (largest variance direction)
    # Since we only have one noise sample, we use the centered feature as direction
    u, s, vt = np.linalg.svd(cls_feat_centered.cpu().float().numpy(), full_matrices=False)
    # u is (1, 1), take the right singular vector vt[0] as the direction
    bias_dirs = vt[:n_dirs].T  # (dim, n_dirs)
    # Normalize
    bias_dirs = bias_dirs / np.linalg.norm(bias_dirs, axis=0, keepdims=True)
    return bias_dirs


def _project_bias(features: torch.Tensor, bias_dirs: torch.Tensor) -> torch.Tensor:
    """Project features to null space of bias directions.

    features: (N, D) — already L2-normalized or raw
    bias_dirs: (D, K) — K orthonormal bias directions
    Returns: (N, D) — features projected to null space, L2-normalized
    """
    # Projection: f' = f - (f @ bias_dirs) @ bias_dirs.T
    proj = features.float() @ bias_dirs  # (N, K)
    features_corrected = features.float() - proj @ bias_dirs.T  # (N, D)
    return F.normalize(features_corrected, p=2, dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5 label-free view-retrieval eval")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--index-csv", type=Path, default=Path("data/processed/_index/index.csv"))
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--n", type=int, default=4096, help="Number of samples in eval set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default=None, help="cuda|cpu (default: auto)")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: next to checkpoint (view_retrieval_step<step>_N<n>.json)",
    )
    ap.add_argument("--topk", type=int, default=5, help="Also compute top-k accuracy")
    ap.add_argument(
        "--ratio", type=float, default=10.0, help="Pass gate: top1 >= ratio*(1/N)"
    )
    ap.add_argument("--scale-aware", action="store_true", help="Enable scale embedding (must match checkpoint)")
    ap.add_argument("--vit-layer", type=int, default=None, help="Extract CLS from this 0-indexed layer instead of final backbone output (arXiv:2604.23670)")
    ap.add_argument("--pos-bias-project", action="store_true", help="Project out positional bias via PCA on noise-image activations (arXiv:2604.23670)")
    args = ap.parse_args()

    if args.pos_bias_project and args.vit_layer is None:
        raise SystemExit("--pos-bias-project requires --vit-layer (need intermediate features for bias computation)")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.index_csv.exists():
        raise FileNotFoundError(f"index_csv not found: {args.index_csv}")
    if not args.split_manifest.exists():
        raise FileNotFoundError(f"split_manifest not found: {args.split_manifest}")
    if args.n <= 0:
        raise SystemExit("--n must be > 0")
    if args.topk <= 0:
        raise SystemExit("--topk must be > 0")

    # Ensure deterministic sampling + view generation.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    m = _load_phase5_module()

    device = None
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load checkpoint
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    step = int(payload.get("step", 0) or 0)
    cfg = payload.get("config", {})
    model_cfg = cfg.get("model", {})

    # Build model (student only)
    if isinstance(model_cfg, m.ModelConfig):
        mc = model_cfg
    else:
        mc = m.ModelConfig(**model_cfg)

    img_size = int(cfg.get("img_size", 224))

    vit = m.PatchViT(
        img_size=img_size,
        patch=mc.patch,
        dim=mc.dim,
        depth=mc.depth,
        heads=mc.heads,
        mlp_ratio=mc.mlp_ratio,
        use_grad_checkpoint=False,
        scale_aware=args.scale_aware,
    )
    student = m.DinoStudentTeacher(vit, out_dim=mc.out_dim).to(device)
    student.load_state_dict(payload["student"], strict=True)
    student.eval()

    # Compute positional bias direction from noise image
    bias_dirs = None
    if args.pos_bias_project:
        print(f"Computing positional bias direction from noise image (layer {args.vit_layer})...")
        bias_dirs_np = _compute_pos_bias_dir(
            student, device, args.vit_layer, mc.dim, img_size=img_size
        )
        bias_dirs = torch.from_numpy(bias_dirs_np).to(device)
        print(f"  Bias direction computed: shape={bias_dirs.shape}")

    # Load rows, filter to val series
    val_set = _load_split_series_dirs(args.split_manifest, "val")
    all_rows = m._load_index_rows(args.index_csv)
    rows = [r for r in all_rows if str(r.series_dir) in val_set]
    if not rows:
        raise SystemExit("No rows remain after filtering to val.series_dir")

    if len(rows) < args.n:
        print(
            f"\u26a0\ufe0f  Requested --n={args.n} but only {len(rows)} val rows available; capping n."  # noqa: T201
        )
        args.n = len(rows)

    rng = random.Random(args.seed)
    idxs = rng.sample(range(len(rows)), k=args.n)

    # Build dataset over all val rows so 3-slice context (z-1,z,z+1) matches training.
    ds = m.PngDataset(
        rows,
        img_size=img_size,
        rw_level_min=float(cfg.get("rw_level_min", -400.0)),
        rw_level_max=float(cfg.get("rw_level_max", 400.0)),
        rw_width_min=float(cfg.get("rw_width_min", 800.0)),
        rw_width_max=float(cfg.get("rw_width_max", 2000.0)),
    )

    # Materialize views + embeddings in batches
    t0 = time.time()
    Q_chunks: list[torch.Tensor] = []
    K_chunks: list[torch.Tensor] = []

    bs = args.batch_size
    for start in range(0, args.n, bs):
        end = min(args.n, start + bs)

        v1_list = []
        v2_list = []
        sp_list = []
        for j in range(start, end):
            views, spacing = ds[idxs[j]]
            v1_list.append(views[0])
            v2_list.append(views[1])
            sp_list.append(spacing)

        x1 = torch.stack(v1_list, dim=0).to(device, non_blocking=True)
        x2 = torch.stack(v2_list, dim=0).to(device, non_blocking=True)
        sp = torch.stack(sp_list, dim=0).to(device, non_blocking=True) if args.scale_aware else None

        layer = args.vit_layer
        q = _embed_backbone_cls(student, x1, spacing=sp, vit_layer=layer)
        k = _embed_backbone_cls(student, x2, spacing=sp, vit_layer=layer)

        if bias_dirs is not None:
            q = _project_bias(q.cpu(), bias_dirs.cpu())
            k = _project_bias(k.cpu(), bias_dirs.cpu())
            Q_chunks.append(q)
            K_chunks.append(k)
        else:
            Q_chunks.append(q.cpu())
            K_chunks.append(k.cpu())

    Q = torch.cat(Q_chunks, dim=0)  # (N, D)
    K = torch.cat(K_chunks, dim=0)  # (N, D)

    # Similarity matrix (N,N)
    # Use float32 on CPU to avoid fp16 weirdness.
    S = (Q.float() @ K.float().T).numpy()

    # Top-1
    top1_idx = np.argmax(S, axis=1)
    top1 = float(np.mean(top1_idx == np.arange(args.n)))

    # Top-k
    k = int(args.topk)
    if k >= args.n:
        k = args.n
    topk_idx = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
    topk = float(np.mean([(i in topk_idx[i]) for i in range(args.n)]))

    baseline = 1.0 / float(args.n)
    ratio = top1 / baseline if baseline > 0 else float("inf")
    passed = top1 >= float(args.ratio) * baseline

    dt = time.time() - t0

    out = args.out
    if out is None:
        suffix = f"_layer{args.vit_layer}" if args.vit_layer is not None else ""
        if args.pos_bias_project:
            suffix += "_biasproj"
        out = args.checkpoint.parent / f"view_retrieval_step{step}_N{args.n}{suffix}.json"

    metrics = {
        "kind": "phase5_view_retrieval",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "step": step,
        "index_csv": str(args.index_csv),
        "split_manifest": str(args.split_manifest),
        "img_size": img_size,
        "n": args.n,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "topk": int(args.topk),
        "top1": top1,
        "topk_acc": topk,
        "random_baseline": baseline,
        "ratio_vs_random": ratio,
        "pass_ratio": float(args.ratio),
        "passed": bool(passed),
        "seconds": dt,
        "model": {
            "name": mc.name,
            "patch": mc.patch,
            "dim": mc.dim,
            "depth": mc.depth,
            "heads": mc.heads,
            "mlp_ratio": mc.mlp_ratio,
            "out_dim": mc.out_dim,
            "ln_out_dim": math.log(float(mc.out_dim)),
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n")

    print("ok=true")
    print(f"passed={str(passed).lower()}")
    print(f"top1={top1:.6f} top{args.topk}={topk:.6f} baseline={baseline:.6f} ratio={ratio:.2f} seconds={dt:.1f}")
    print(f"metrics_json={out}")

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
