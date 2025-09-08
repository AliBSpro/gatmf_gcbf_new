#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys

# 允许从项目根目录调用
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from convert.convert_model import pick_latest_pair
from convert.pytorch_to_jax_converter import PyTorchToJAXConverter, save_flax_params

def main():
    parser = argparse.ArgumentParser(
        description="Convert latest GAT-MF PyTorch checkpoints to Flax params for GCBF+."
    )
    parser.add_argument(
        "--model_root", type=str, default="./model",
        help="Root folder containing training outputs (subfolders with actor_*.pth etc.). Default: ./model"
    )
    parser.add_argument(
        "--n_agents", type=int, required=True,
        help="Number of agents used during training (must match your GAT-MF config)."
    )
    parser.add_argument(
        "--save_name", type=str, default="gat_mf_flax.pkl",
        help="Output filename (saved into the chosen model subfolder). Default: gat_mf_flax.pkl"
    )
    parser.add_argument(
        "--subdir", type=str, default=None,
        help="Optional: specify a particular subfolder inside model_root (e.g. 6grid_2025-09-08_15-30-12). "
             "If not given, the latest subfolder will be used automatically."
    )
    parser.add_argument(
        "--step", type=int, default=None,
        help="Optional: force a particular training step to pick actor_XX.pth / actor_attention_XX.pth. "
             "If not provided, the latest pair will be selected."
    )
    args = parser.parse_args()

    model_root = os.path.abspath(args.model_root)
    if not os.path.isdir(model_root):
        print(f"[ERROR] model_root not found: {model_root}")
        sys.exit(1)

    # auto / manual subdir
    if args.subdir:
        latest_dir = os.path.join(model_root, args.subdir)
        if not os.path.isdir(latest_dir):
            print(f"[ERROR] subdir not found: {latest_dir}")
            sys.exit(1)
        # 用 pick_latest_pair 的逻辑，但限定在该子目录
        actor_path, attn_path, latest_dir, step = pick_latest_pair(latest_dir)
    else:
        # 在整个 model_root 下挑选
        actor_path, attn_path, latest_dir, step = pick_latest_pair(model_root)

    # 如果用户强制指定了 step，则把路径替换为该 step 的文件（若存在）
    if args.step is not None:
        step_tag = str(args.step)
        cand_actor = os.path.join(latest_dir, f"actor_{step_tag}.pth")
        cand_attn  = os.path.join(latest_dir, f"actor_attention_{step_tag}.pth")
        if os.path.isfile(cand_actor) and os.path.isfile(cand_attn):
            actor_path, attn_path, step = cand_actor, cand_attn, args.step
        else:
            print(f"[WARN] step={args.step} not found in {latest_dir}, "
                  f"fallback to latest pair at step={step}.")

    print("=== Converting GAT-MF checkpoints to Flax params ===")
    print(f"Model root : {model_root}")
    print(f"Chosen dir : {latest_dir}")
    print(f"Actor      : {actor_path}")
    print(f"Attention  : {attn_path}")
    print(f"n_agents   : {args.n_agents}")

    converter = PyTorchToJAXConverter()
    flax_params = converter.convert_pytorch_to_jax(
        pytorch_actor_path=actor_path,
        pytorch_attention_path=attn_path,
        n_agents=args.n_agents,
    )

    save_path = os.path.join(latest_dir, args.save_name)
    save_flax_params(flax_params, save_path)
    print(f"[OK] Saved Flax params -> {save_path}")

if __name__ == "__main__":
    main()
