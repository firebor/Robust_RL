# Author: Loren
# Date: 2025-03-08
#
# This script is used to generate plots from npy files.
#
# Usage: python draw_from_npy.py <path_to_npy_dir_or_parent>

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.signal import savgol_filter


def load_npy_files(npy_dir):
    """Load all npy files from npy directory"""
    npy_files = sorted(npy_dir.rglob("*.npy"))
    if not npy_files:
        raise ValueError(f"No npy files found in {npy_dir}")
    
    all_data = []
    for npy_file in tqdm(npy_files, desc="📂 Loading npy files"):
        try:
            data = np.load(npy_file, allow_pickle=True).item()
            all_data.append(data)
        except Exception as e:
            print(f"⚠️ Failed to load {npy_file}: {str(e)}")
    
    return all_data


def generate_plots_from_npy(npy_dir, smooth=True, smooth_window=51, smooth_polyorder=3):
    """Generate plots from npy files"""
    fig_out_dir = npy_dir.parent / "plots_from_npy"
    fig_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Plots will be saved to {fig_out_dir}")
    
    all_data = load_npy_files(npy_dir)
    
    # 收集所有tag的数据
    tag_data = {}
    for data in all_data:
        meta = data["meta"]
        for tag, tag_data_dict in data["data"].items():
            if tag not in tag_data:
                tag_data[tag] = []
            
            tag_data[tag].append({
                "steps": tag_data_dict["steps"],
                "values": tag_data_dict["values"],
                "name": meta["run"],
                "env": meta["env"],
                "exp": meta["exp"]
            })
    
    # 绘制每个tag的图表
    for tag, data_list in tqdm(tag_data.items(), desc="🎨 Generating plots"):
        plt.figure(figsize=(12, 7))
        
        for data in data_list:
            # 绘制原始数据(半透明)
            plt.plot(
                data["steps"], 
                data["values"], 
                alpha=0.2, 
                linewidth=1,
                # label=f"{data['name']}_{data['exp'].capitalize()} (raw)"
            )
            
            # 绘制平滑数据(实线)
            if smooth and len(data["values"]) >= smooth_window:
                smoothed_values = savgol_filter(data["values"], smooth_window, smooth_polyorder)
                plt.plot(
                    data["steps"], 
                    smoothed_values, 
                    alpha=1.0, 
                    linewidth=1,
                    label=f"{data['name']}_{data['exp'].capitalize()}",
                    color=plt.gca().lines[-1].get_color(),
                )
        
        # 图表装饰
        plt.title(f"{data_list[0]['env']} - {data_list[0]['exp']} - {tag.split('/')[-1].title()}")
        plt.xlabel("Training Steps", fontsize=12)
        plt.ylabel(tag.split("/")[-1].title(), fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 智能图例布局
        handles, labels = plt.gca().get_legend_handles_labels()
        ncol = 2 if len(labels) > 8 else 1
        plt.legend(
            handles=handles,
            labels=labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            ncol=ncol,
            frameon=False
        )
        
        # 保存图表
        save_path = fig_out_dir / f"{tag.split('/')[-1]}.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()


def main(target_path: str):
    log_dir = Path(target_path).resolve()
    print("🚀 Processing log path:", log_dir)
    
    # 查找所有npy_data目录
    for npy_dir in log_dir.rglob("npy_data"):
        print(f"\n📂 Found npy directory: {npy_dir}")
        generate_plots_from_npy(npy_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python draw_from_npy.py <path_to_npy_dir_or_parent>")
        sys.exit(1)
    
    npy_dir = sys.argv[1]
    # npy_dir = "/home/liuhongbo/workspace/Robust_RL/SAC/train_result/small_batch_train/Walker2d-v2"
    main(npy_dir)
