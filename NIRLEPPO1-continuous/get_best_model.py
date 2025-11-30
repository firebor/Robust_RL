#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIPPO算法最佳模型提取工具

从训练结果中提取每个种子下表现最好的模型，并复制到best_model目录
模型文件格式为 agent_xxx_yyyy，其中 xxx 表示训练步数，yyyy 表示奖励值
"""

import os
import re
import shutil
import argparse
from pathlib import Path


def get_best_model(model_name='Walker2d-v2', exp_name='RLE', min_steps=0):
    """
    在指定目录下查找所有子文件夹中效果最好的模型
    
    Args:
        model_name: 环境名称
        exp_name: 实验名称
        min_steps: 最小训练步数要求（可选，用于过滤早期模型）
    
    Returns:
        extracted_count: 成功提取的模型数量
    """
    model_dir = f'./train_result/{exp_name}/{model_name}/model_weights'
    
    if not os.path.exists(model_dir):
        print(f"错误：目录 {model_dir} 不存在")
        print("请确保：")
        print(f"1. 实验名称正确: {exp_name}")
        print(f"2. 环境名称正确: {model_name}")
        print("3. 已经完成了模型训练")
        return 0
    
    # 创建best_model目录
    best_model_dir = f'./best_model/{exp_name}/{model_name}'
    os.makedirs(best_model_dir, exist_ok=True)
    print(f"创建目标目录: {best_model_dir}")
    
    extracted_count = 0
    
    # 遍历主目录下的所有子目录（每个种子对应一个子目录）
    for subdir in os.listdir(model_dir):
        subdir_path = os.path.join(model_dir, subdir)
        
        # 确保是目录
        if not os.path.isdir(subdir_path):
            continue
        
        print(f"\n处理目录: {subdir}")
        
        # 创建目标子目录
        target_subdir_path = os.path.join(best_model_dir, subdir)
        os.makedirs(target_subdir_path, exist_ok=True)
        
        # 在子目录中查找最佳模型
        best_model_file = None
        best_reward = -float('inf')
        best_steps = 0
        
        model_files = []
        
        for file in os.listdir(subdir_path):
            # 匹配NIPPO模型文件名格式 agent_xxx_yyyy
            # xxx: 训练步数, yyyy: 奖励值
            match = re.match(r'agent_(\d+)_([+-]?\d+(?:\.\d+)?)', file)
            if match:
                steps = int(match.group(1))
                reward = float(match.group(2))
                model_files.append((file, steps, reward))
        
        if not model_files:
            print(f"  警告：在 {subdir} 中没有找到有效的模型文件")
            continue
        
        # 过滤满足最小步数要求的模型
        valid_models = [(f, s, r) for f, s, r in model_files if s >= min_steps]
        
        if not valid_models:
            print(f"  警告：在 {subdir} 中没有找到满足最小步数要求({min_steps})的模型")
            continue
        
        # 找到奖励最高的模型
        best_model_file, best_steps, best_reward = max(valid_models, key=lambda x: x[2])
        
        print(f"  找到最佳模型: {best_model_file}")
        print(f"  训练步数: {best_steps}")
        print(f"  奖励值: {best_reward}")
        
        # 复制最佳模型到目标目录
        source_path = os.path.join(subdir_path, best_model_file)
        target_path = os.path.join(target_subdir_path, best_model_file)
        
        try:
            shutil.copy2(source_path, target_path)
            print(f"  ✓ 已复制到: {target_path}")
            extracted_count += 1
        except Exception as e:
            print(f"  ✗ 复制失败: {e}")
    
    print(f"\n==================")
    print(f"提取完成！共处理 {extracted_count} 个最佳模型")
    print(f"结果保存在: {best_model_dir}")
    print(f"==================")
    
    return extracted_count


def batch_extract_best_models(exp_name='PPO-continuous', env_list=None, min_steps=0):
    """
    批量提取多个环境的最佳模型
    
    Args:
        exp_name: 实验名称
        env_list: 环境列表，如果为None则自动检测
        min_steps: 最小训练步数要求
    """
    train_result_dir = f'./train_result/{exp_name}'
    
    if not os.path.exists(train_result_dir):
        print(f"错误：实验目录 {train_result_dir} 不存在")
        return
    
    # 如果没有指定环境列表，则自动检测
    if env_list is None:
        env_list = []
        for item in os.listdir(train_result_dir):
            item_path = os.path.join(train_result_dir, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'model_weights')):
                env_list.append(item)
        
        if not env_list:
            print(f"在 {train_result_dir} 中没有找到有效的环境目录")
            return
    
    print(f"开始批量提取最佳模型...")
    print(f"实验名称: {exp_name}")
    print(f"环境列表: {env_list}")
    print(f"最小步数: {min_steps}")
    print("=" * 50)
    
    total_extracted = 0
    
    for env_name in env_list:
        print(f"\n处理环境: {env_name}")
        print("-" * 30)
        
        extracted = get_best_model(env_name, exp_name, min_steps)
        total_extracted += extracted
    
    print("\n" + "=" * 50)
    print(f"批量提取完成！总共提取了 {total_extracted} 个最佳模型")


def main():
    parser = argparse.ArgumentParser(description='NIPPO算法最佳模型提取工具')
    
    parser.add_argument('--env_name', type=str, default='Walker2d-v2',
                       help='环境名称 (默认: Walker2d-v2)')
    parser.add_argument('--exp_name', type=str, default='RLE',
                       help='实验名称 (默认: PPO-continuous)')
    parser.add_argument('--min_steps', type=int, default=0,
                       help='最小训练步数要求 (默认: 0)')
    parser.add_argument('--batch', action='store_true',
                       help='批量处理模式，提取所有环境的最佳模型')
    parser.add_argument('--env_list', nargs='+', default=None,
                       help='批量模式下的环境列表 (可选)')
    
    args = parser.parse_args()
    
    print("NIPPO算法最佳模型提取工具")
    print("=" * 40)
    
    if args.batch:
        # 批量提取模式
        batch_extract_best_models(args.exp_name, args.env_list, args.min_steps)
    else:
        # 单个环境提取模式
        get_best_model(args.env_name, args.exp_name, args.min_steps)


if __name__ == "__main__":
    main() 