import os, re, glob

def pick_latest_pair(model_root: str):
    if not os.path.isdir(model_root):
        raise FileNotFoundError(f"模型根目录不存在: {model_root}")

    # 1) 选最新子目录
    subdirs = [os.path.join(model_root, d) for d in os.listdir(model_root)
               if os.path.isdir(os.path.join(model_root, d))]
    if not subdirs:
        raise FileNotFoundError(f"{model_root} 下没有任何子目录（没有训练产物）")
    subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    latest_dir = subdirs[0]

    # 2) 读取这个目录里的文件列表
    all_pths = glob.glob(os.path.join(latest_dir, "*.pth"))

    # 用正则区分：actor_* vs actor_attention_*
    # actor: 以 "actor_" 开头，但后面不是 "attention_"
    actor_paths = [p for p in all_pths
                   if re.search(r"/actor_(?!attention_).+\.pth$", p)]
    attn_paths  = [p for p in all_pths
                   if re.search(r"/actor_attention_.+\.pth$", p)]

    if not actor_paths:
        raise FileNotFoundError(f"{latest_dir} 里没有 actor_*.pth（不含 attention 的那种）")
    if not attn_paths:
        raise FileNotFoundError(f"{latest_dir} 里没有 actor_attention_*.pth")

    def extract_step(path):
        m = re.search(r"_(\d+)\.pth$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    actors_by_step = {extract_step(p): p for p in actor_paths if extract_step(p) >= 0}
    attns_by_step  = {extract_step(p): p for p in attn_paths if extract_step(p) >= 0}

    common = sorted(set(actors_by_step) & set(attns_by_step))
    if not common:
        raise RuntimeError(
            f"在 {latest_dir} 找不到步数一致的一对权重。\n"
            f"actor 步数: {sorted(actors_by_step)}\n"
            f"attention 步数: {sorted(attns_by_step)}"
        )
    step = common[-1]
    return actors_by_step[step], attns_by_step[step], latest_dir, step
