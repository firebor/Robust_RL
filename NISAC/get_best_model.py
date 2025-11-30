# 输入环境

import os
import re

def get_best_model(model_name='Hopper-v2',exp_name='polynomial_grad_norm_scaled_decay'):
    """
    在指定目录下查找所有子文件夹中效果最好的模型
    模型文件格式为 agent_xxx_yyyy，其中 yyyy 表示奖励值，越大越好
    
    Args:
        model_dir: 模型文件夹路径
    
    Returns:
        best_model_path: 最佳模型的完整路径
        best_reward: 最佳模型的奖励值
    """
    model_dir='./train_result/'+exp_name+'/'+model_name+'/model_weights'
    if not os.path.exists(model_dir):
        print(f"错误：目录 {model_dir} 不存在")
        return None, 0
    
    os.makedirs(f'./best_model/{exp_name}/{model_name}', exist_ok=True)

    # 遍历主目录下的所有子目录
    for subdir in os.listdir(model_dir):
        # 创建目标目录结构，但不复制文件
        target_subdir_path = os.path.join(f'./best_model/{exp_name}/{model_name}', subdir)
        if not os.path.exists(target_subdir_path):
            os.makedirs(target_subdir_path, exist_ok=True)
        subdir_path = os.path.join(model_dir, subdir)
        
        # 确保是目录
        if os.path.isdir(subdir_path):
            # 遍历子目录中的所有文件
            subdir_best_model = None
            subdir_best_reward = -float('inf')
            subdir_best_model_path = None
            
            for file in os.listdir(subdir_path):
                # 匹配模型文件名格式 agent_xxx_yyyy
                match = re.match(r'agent_(\d+)_(\d+)', file)
                if match:
                    episode = int(match.group(1))
                    reward = int(match.group(2))
                    
                    # 更新子目录中的最佳模型
                    # if reward > subdir_best_reward and episode > 800000:
                    if reward > subdir_best_reward:
                        subdir_best_reward = reward
                        subdir_best_model = file
                        subdir_best_model_path = os.path.join(subdir_path, file)
            
            # 将子目录中的最佳模型复制到目标目录
            if subdir_best_model:
                import shutil
                target_model_path = os.path.join(target_subdir_path, subdir_best_model)
                shutil.copy2(subdir_best_model_path, target_model_path)
                print(f"已将最佳模型 {subdir_best_model} 从 {subdir_path} 复制到 {target_subdir_path}")
                
    
    return

if __name__ == "__main__":
    get_best_model()
