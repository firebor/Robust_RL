# 画折线图 直接 is_draw_zhexian=True 画折线图 其他false,然后algorithm_params添加key-value选择算法

from draw import draw_per_seed_comparison_lines # 更新导入名称
from heatmap_Y import draw_heatmap
from draw_msr import draw_msr #,draw_msr_with_seed # 移除 with_seed导入
from draw_msr_norm import draw_msr_norm #,draw_msr_norm_with_seed # 移除 with_seed导入
from surface_html import draw_surface_html
from surface_png import draw_surface_png
import numpy as np
import os # 导入os以创建根目录

# --- 配置区 --- 

# 根目录创建 (确保 './res' 存在)
os.makedirs('./res', exist_ok=True)

# 环境名称
env_name = 'HalfCheetah-v2' 


# 算法参数配置
# 每个算法一个字典项，包含:
#   'exp-name': 实验名称 (用于构建路径)
#   'seed': 该算法使用的种子列表 (list of int)
#   'noise': 噪声值 (float or None)，如果为 None 或键不存在，则加载不带 noise 后缀的文件
env_name = 'HalfCheetah-v2' 
algorithm_params = {        
    'SAC': {'exp-name': 'Adam-optimizer_cross',
            'seed': [66, 123, 7],
            'noise': None},
    # 'SCSAC': {'exp-name': 'Adam-optimizer_cross',
    #          'seed': [66, 123, 7],
    #          'noise': None},
    # 'NISAC': {'exp-name': 'polynomial_grad_norm_scaled_decay_cross',
    #          'seed': [66, 123, 7],
    #          'noise': 1.0},
}

env_name = 'Hopper-v2' 
algorithm_params = {        
    'SAC': {'exp-name': 'Adam-optimizer_cross',
            'seed': [7, 100, 2025],
            'noise': None},
    'SCSAC': {'exp-name': 'Adam-optimizer_cross',
             'seed': [73, 314, 100],
             'noise': None},
    'NISAC': {'exp-name': 'polynomial_grad_norm_scaled_decay_cross',
             'seed': [42, 123, 2025],
             'noise': 1.0},
}

env_name = 'Walker2d-v2' 
algorithm_params = {        
    # 'SAC': {'exp-name': 'Adam-optimizer_cross',
    #         'seed': [0, 314, 999],
    #         'noise': None},
    # 'SCSAC': {'exp-name': 'Adam-optimizer_cross',
    #          'seed': [66, 314, 2025],
    #          'noise': None},
    # 'NISAC': {'exp-name': 'norm_scaled_noise_cross',
    #          'seed': [7, 66, 73, 100],
    #          'noise': 1.0},
    'NIRLEPPO1-continuous': {'exp-name': 'RLE_cross',
             'seed': [42],
             'noise': None,
             'sw': 70},
    'NIPPO-continuous': {'exp-name': 'PPO-continuous_cross',
            'seed': [42],
            'noise': None,
            'sw': None},
}

# 选择要绘制的图表类型
is_draw_per_seed_lines = False   # 折线图 (每个种子一张对比图)
is_draw_msr = False            # 残影图 (Mean ± Std) (一张对比图)
is_draw_msr_norm = False     # 归一化残影图 (Mean ± Std Norm) (一张对比图)
is_draw_heatmap = True        # 热力图 (每个算法每个种子一张图)
is_draw_surface_html = False   # 3D曲面图 HTML (每个算法每个种子一张图)
is_draw_surface_png = False  # 3D曲面图 PNG (每个算法每个种子一张图)

# --- 执行区 --- 

if __name__ == "__main__":
    print("===== 开始执行绘图任务 ====")
    print(f"环境: {env_name}")
    print(f"算法参数配置: {algorithm_params}")
    print("---------------------------")

    if is_draw_per_seed_lines:
        # 调用更新后的函数名
        draw_per_seed_comparison_lines(env_name, algorithm_params)
        print("---------------------------")

    if is_draw_msr:
        # draw_msr 现在只需要 env_name 和 algorithm_params
        draw_msr(env_name, algorithm_params)
        print("---------------------------")

    if is_draw_msr_norm:
        # draw_msr_norm 现在只需要 env_name 和 algorithm_params
        draw_msr_norm(env_name, algorithm_params)
        print("---------------------------")

    if is_draw_heatmap:
        # draw_heatmap 现在只需要 env_name 和 algorithm_params
        # draw_heatmap(env_name, algorithm_params)
        draw_heatmap(env_name, algorithm_params, vmin=0, vmax=6000)
        print("---------------------------")

    if is_draw_surface_html:
        # draw_surface_html 现在只需要 env_name 和 algorithm_params
        draw_surface_html(env_name, algorithm_params)
        print("---------------------------")

    if is_draw_surface_png:
        # draw_surface_png 现在只需要 env_name 和 algorithm_params
        draw_surface_png(env_name, algorithm_params)
        print("---------------------------")

    print("===== 所有绘图任务执行完毕 ====")


    
