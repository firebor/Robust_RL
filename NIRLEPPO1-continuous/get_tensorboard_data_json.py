#  AIGC START
"""
从TensorBoard日志中导出数据到JSON格式
支持从event文件中读取scalar数据并导出为JSON
导出格式: [[wall_time, step, value], ...]
文件名格式: {env_name}_{algo}_seed{seed}.json
"""
import os
import json
import argparse
import re
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_scalars_from_tensorboard(log_dir, tag_name=None):
    """
    从TensorBoard日志目录中提取scalar数据
    
    Args:
        log_dir: TensorBoard日志目录路径
        tag_name: 要提取的tag名称，如果为None则提取第一个tag
        
    Returns:
        list: 格式为 [[wall_time, step, value], ...] 的列表
    """
    # 创建EventAccumulator
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    # 获取所有scalar tags
    scalar_tags = ea.Tags()['scalars']
    
    if not scalar_tags:
        return None, None
    
    # 选择要导出的tag
    if tag_name is None:
        # 如果没有指定tag，使用第一个tag
        tag = scalar_tags[0]
    else:
        if tag_name not in scalar_tags:
            print(f"警告: tag '{tag_name}' 不存在，使用第一个tag '{scalar_tags[0]}'")
            tag = scalar_tags[0]
        else:
            tag = tag_name
    
    # 提取scalar数据
    scalar_events = ea.Scalars(tag)
    data = [
        [float(event.wall_time), int(event.step), float(event.value)]
        for event in scalar_events
    ]
    
    return data, tag


def parse_log_dir_name(log_dir):
    """
    从日志目录路径中解析env_name, algo, seed
    
    Args:
        log_dir: TensorBoard日志目录路径
        
    Returns:
        tuple: (env_name, algo, seed) 或 (None, None, None)
    """
    log_path = Path(log_dir)
    dir_name = log_path.name
    
    # 首先尝试从路径中获取env_name
    # 路径格式通常是: .../train_result/{exp_name}/{env_name}/tensorboard_logs/{dir_name}
    env_name = None
    parts = log_path.parts
    if 'tensorboard_logs' in parts:
        idx = parts.index('tensorboard_logs')
        if idx > 0:
            env_name = parts[idx - 1]
    
    # 从目录名中提取algo和seed
    # 格式: {env_name}_{algo}_seed{seed} 或 {algo}_seed{seed} 或 {env_name}_{algo}_seed{seed}_*
    # 例如: InvertedDoublePendulum-v2_sac_seed5571
    # 匹配最后一个 _seed{数字} 模式
    match = re.search(r'_([^_]+)_seed(\d+)', dir_name)
    if match:
        algo = match.group(1)
        seed = match.group(2)
        
        # 如果从路径中获取了env_name，直接使用
        if env_name:
            return env_name, algo, seed
        
        # 否则尝试从目录名中提取env_name（匹配到最后一个_algo_seed之前的部分）
        # 例如: InvertedDoublePendulum-v2_sac_seed5571 -> env_name = InvertedDoublePendulum-v2
        env_match = re.match(r'(.+?)_' + re.escape(algo) + r'_seed' + re.escape(seed), dir_name)
        if env_match:
            env_name = env_match.group(1)
            return env_name, algo, seed
    
    return None, None, None


def generate_output_filename(log_dir, tag_name=None, output_dir=None):
    """
    生成输出文件名
    
    Args:
        log_dir: TensorBoard日志目录路径
        tag_name: tag名称（可选）
        output_dir: 输出目录（可选）
        
    Returns:
        str: 输出文件路径
    """
    env_name, algo, seed = parse_log_dir_name(log_dir)
    
    if env_name and algo and seed:
        filename = f"{env_name}_{algo}_seed{seed}.json"
    else:
        # 如果无法解析，使用目录名
        log_path = Path(log_dir)
        filename = f"{log_path.name}.json"
        print(f"警告: 无法从路径解析信息，使用默认文件名: {filename}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)
    else:
        log_path = Path(log_dir)
        return str(log_path.parent / filename)


def export_to_json(data, output_path):
    """
    将数据导出为JSON格式
    
    Args:
        data: 格式为 [[wall_time, step, value], ...] 的列表
        output_path: 输出JSON文件路径
    """
    # 保存为JSON（不缩进，节省空间）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    
    print(f"数据已导出到: {output_path}")
    print(f"共导出 {len(data)} 个数据点")


def find_seed_directories(tensorboard_logs_dir):
    """
    在tensorboard_logs目录下查找所有种子目录
    
    Args:
        tensorboard_logs_dir: tensorboard_logs目录路径
        
    Returns:
        list: 所有种子目录的路径列表
    """
    seed_dirs = []
    tb_logs_path = Path(tensorboard_logs_dir)
    
    if not tb_logs_path.exists():
        return seed_dirs
    
    # 遍历tensorboard_logs下的所有子目录
    for item in tb_logs_path.iterdir():
        if item.is_dir():
            # 检查目录中是否有event文件
            event_files = list(item.glob('events.out.tfevents.*'))
            if event_files:
                seed_dirs.append(str(item))
    
    return sorted(seed_dirs)

# python get_tensorboard_data_json.py --exp_name RLE --env_name Walker2d-v2
def main():
    parser = argparse.ArgumentParser(description='从TensorBoard日志导出数据到JSON')
    parser.add_argument('--exp_name', type=str, required=True,
                       help='实验名称，例如: default-exp')
    parser.add_argument('--env_name', type=str, required=True,
                       help='环境名称，例如: InvertedDoublePendulum-v2')
    parser.add_argument('--tag', type=str, default=None,
                       help='要导出的tag名称（如果未指定，使用第一个tag）')
    parser.add_argument('--train_result_dir', type=str, default='./train_result',
                       help='train_result目录路径（默认: ./train_result）')
    parser.add_argument('--output_base_dir', type=str, default='../draw_converg/json_file/RLE',
                       help='JSON输出基础目录（默认: ./json_file）')
    
    args = parser.parse_args()
    
    # 构建tensorboard_logs目录路径
    tensorboard_logs_dir = Path(args.train_result_dir) / args.exp_name / args.env_name / 'tensorboard_logs'
    
    if not tensorboard_logs_dir.exists():
        print(f"错误: tensorboard_logs目录不存在: {tensorboard_logs_dir}")
        return
    
    # 查找所有种子目录
    seed_dirs = find_seed_directories(tensorboard_logs_dir)
    
    if not seed_dirs:
        print(f"警告: 在 {tensorboard_logs_dir} 下未找到任何包含event文件的目录")
        return
    
    print(f"找到 {len(seed_dirs)} 个种子目录")
    
    # 创建输出目录
    output_dir = Path(args.output_base_dir) / args.env_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 处理每个种子目录
    success_count = 0
    for seed_dir in seed_dirs:
        print(f"\n处理: {seed_dir}")
        try:
            data, tag = extract_scalars_from_tensorboard(seed_dir, args.tag)
            if data is not None and len(data) > 0:
                # 生成输出文件名
                output_path = generate_output_filename(seed_dir, tag, str(output_dir))
                
                export_to_json(data, output_path)
                if tag:
                    print(f"  导出的tag: {tag}")
                success_count += 1
            else:
                print(f"  警告: {seed_dir} 中没有找到scalar数据")
        except Exception as e:
            print(f"  错误: 处理 {seed_dir} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n完成! 成功导出 {success_count}/{len(seed_dirs)} 个文件到 {output_dir}")


# python get_tensorboard_data_json.py --exp_name default-exp --env_name InvertedDoublePendulum-v2
if __name__ == '__main__':
    main()


