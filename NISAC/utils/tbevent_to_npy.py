# Author: Loren
# Date: 2025-03-08
#
# This script is used to convert tensorboard event files to numpy files.
# The numpy files will be saved in the same directory as the tensorboard event files.
# Usage: python tbevent_to_npy.py <path_to_tensorboard_logs>

import sys
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm.auto import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_single_file(tfevent_file: Path, progress_queue=None, npy_dir=None):
    """处理单个文件并提取元数据"""
    try:
        # 加载事件文件
        event_acc = EventAccumulator(str(tfevent_file))
        event_acc.Reload()
        
        # 统一提取元数据
        path_parts = tfevent_file.parent.parts
        run_name = path_parts[-1]
        env_name = path_parts[-3]
        exp_name = path_parts[-4]
        algorithm_name = path_parts[-6]
        
        result = {
            "meta": {
                "env": env_name,
                "exp": exp_name,
                "run": run_name,
                "algorithm": algorithm_name,
                "file": tfevent_file.name
            },
            "data": {}
        }
        
        # 提取数据
        for tag in event_acc.Tags()["scalars"]:
            events = event_acc.Scalars(tag)
            result["data"][tag] = {
                "steps": np.array([e.step for e in events]),
                "values": np.array([e.value for e in events])
            }
        
        # 保存npy文件
        if npy_dir:
            save_path = npy_dir / f"{tfevent_file.stem}.npy"
            np.save(save_path, np.array(result, dtype=object))
        
        # 更新进度
        if progress_queue:
            progress_queue.put(1)
        return (True, run_name, result)
    
    except Exception as e:
        if progress_queue:
            progress_queue.put(1)
        return (False, str(tfevent_file), str(e))
    

def parallel_converter(tfevent_files, max_workers=None, npy_dir=None):
    """并行转换主逻辑"""
    total_files = len(tfevent_files)
    
    ctx = mp.get_context('spawn')
    progress_queue = ctx.Queue()
    
    with tqdm(total=total_files, desc="🚀 Converting Files", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers or mp.cpu_count()) as executor:
            # 提交所有任务
            futures = {
                executor.submit(load_single_file, f, progress_queue, npy_dir): f
                for f in tfevent_files
            }

            # 实时更新进度条
            def update_progress():
                while pbar.n < total_files:
                    try:
                        progress_queue.get(timeout=1)
                        pbar.update()
                    except:
                        continue

            # 启动进度更新线程
            from threading import Thread
            progress_thread = Thread(target=update_progress)
            progress_thread.start()

            # 等待所有任务完成
            for future in as_completed(futures):
                success, identifier, data = future.result()
                if not success:
                    tqdm.write(f"⚠️ Failed {identifier}: {data}")

            # 等待进度更新线程完成
            progress_thread.join()
            

def main(target_path: str):
    # Will iter every tfwvent file in target_path and its subdirectories.
    log_dir = Path(target_path).resolve()
    if log_dir.name == "tensorboard_logs":
        log_dir = log_dir.parent
        print("Warning: tensorboard_logs is not a valid Experiment path, using its parent directory.")
    print("🚀 Processing log path:", log_dir)
    
    # for every tensorboard_logs directory
    for tb_log_dir in log_dir.rglob("tensorboard_logs"):
        npy_dir = tb_log_dir.parent / "npy_data"
        npy_dir.mkdir(parents=True, exist_ok=True)
        
        tfevent_files = sorted(tb_log_dir.rglob("events.out.tfevents.*"))
        print(f"📂 Found {len(tfevent_files)} files in {tb_log_dir}")
        
        if tfevent_files:
            # Convert tensorboard event files to numpy files.
            parallel_converter(tfevent_files, npy_dir=npy_dir)
            print(f"✅ Converted {len(tfevent_files)} files to {npy_dir}")
            

if __name__ == "__main__":
    # tb_logdir = "/home/liuhongbo/workspace/Robust_RL/PDSAC/train_result/grad-norm/Walker2d-v2"
    # tb_logdir = "/home/liuhongbo/workspace/Robust_RL/SAC/train_result"
    if len(sys.argv) < 2:
        print("Usage: python tbevent_to_npy.py <path_to_tensorboard_logs>")
        exit(1)
        
    tb_logdir = sys.argv[1]
    main(tb_logdir)
