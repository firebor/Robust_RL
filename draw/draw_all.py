# 画折线图 直接 is_draw_zhexian=True 画折线图 其他false,然后algorithm_params添加key-value选择算法

from draw import draw_per_seed_comparison_lines # 更新导入名称
from heatmap import draw_heatmap
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
    # 'SAC': {'exp-name': 'default-exp_cross',
    #         'seed': [42, 123, 0, 2025, 7, 100, 999, 314, 73, 66, 61209, 9834, 77025, 14563, 30298, 5571, 84602, 21937, 69840, 10526, 43781, 96215, 2804, 71359, 52468, 8901, 63724, 17890, 94532, 36107, 5824, 75619, 20483, 67390, 1125, 82976, 40158, 95723, 26431, 7806, 59342, 13874, 86509, 32167, 4790, 70258, 25613, 91470, 6482, 38795, 53061, 19247, 85630, 4108, 76924, 29571, 6038, 97145, 33620, 5489, 81276, 16039, 72584, 23910, 68472, 1075, 49361, 88027, 35492, 5718, 79603, 22845, 65170, 18396, 90214, 44753, 3106, 73829, 27158, 5640, 83971, 15284, 62739, 9406, 48512, 37690, 5083, 86147, 24905, 71268, 13579, 99999, 42069, 66666, 10001, 88888, 55555, 22222, 77777, 33333],
    #         'noise': None},
    # 'SCSAC': {'exp-name': 'Adam-optimizer_cross',
    #          'seed': [66, 314, 2025],
    #          'noise': None},
    # 'NISAC': {'exp-name': 'norm_scaled_noise_cross',
    #          'seed': [7, 66, 73, 100],
    #          'noise': 1.0},
    'NIRLEPPO1-continuous': {'exp-name': 'RLE_cross',
             'seed': [42, 123, 0, 2025, 7, 100, 999, 314, 73, 66],
             'noise': None,
             'sw': 70},
    'NIPPO-continuous': {'exp-name': 'PPO-continuous_cross',
            'seed': [42, 123, 0, 2025, 7, 100, 999, 314, 73, 66],
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
        draw_heatmap(env_name, algorithm_params, vmin=0, vmax=6500)
        # draw_heatmap(env_name, algorithm_params)
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


    
