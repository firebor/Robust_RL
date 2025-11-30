import numpy as np
import plotly.graph_objects as go
import os

# seeds = [73]
# seeds=[42,123,0,2025,7,100,999,73,66,314]
seeds=[0, 42, 123, 2025]
algorithm= "NISAC"
noise = 1.0
exp_name = 'NISAC_sigmoid_1_grad_norm_cross'
env_name = 'Walker2d-v2'
file_path=''
if noise:
    file_path=f'./res/surfacemap/{algorithm}_n{noise}/{exp_name}/{env_name}/html'
else:
    file_path=f'./res/surfacemap/{algorithm}/{exp_name}/{env_name}/html'
os.makedirs(file_path, exist_ok=True)

for seed in seeds:
    if noise:
        reward_pdsac = np.load(f'../{algorithm}/test_result/{exp_name}/{env_name}/result_numpy/{env_name}_seed{seed}_noise{noise}.npy')
    else:
        reward_pdsac = np.load(f'../{algorithm}/test_result/{exp_name}/{env_name}/result_numpy/{env_name}_seed{seed}.npy')
    # reward_pdsac = reward_pdsac[4:15, 4:15]  # 11x11 数据
    
    # 生成网格坐标（-0.5~0.5）
    x = np.linspace(0.5, 1.5, 51)
    y = np.linspace(0.5, 1.5, 51)
    X, Y = np.meshgrid(x, y)

    custom_colorscale = [
        [0.0, 'rgb(0,0,255)'],    # 蓝色在最小值
        [0.5, 'rgb(215,220,226)'], # 白色在中值
        [1.0, 'rgb(180,4,38)']      # 红色在最大值
    ]
    coolwarm_plotly = [
        [0.0, 'rgb(59, 76, 192)'],    # 冷色 (深蓝)
        [0.5, 'rgb(221, 221, 221)'],  # 中性 (灰)
        [1.0, 'rgb(180, 4, 38)']      # 暖色 (深红)
    ]
    # 创建交互式3D曲面图
    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=reward_pdsac,
        colorscale=coolwarm_plotly,
        showscale=True,
        colorbar=dict(title='Reward')
    )])

    # 更新布局
    fig.update_layout(
        title=f'3D Reward Surface (Seed={seed})',
        scene=dict(
            xaxis_title='Friction',
            yaxis_title='Mass',
            zaxis_title='Episode Reward',
            xaxis=dict(
                ticktext=[f'{x:.1f}' for x in np.linspace(0.5, 1.5, 11)],
                tickvals=np.linspace(0.5, 1.5, 11)
            ),
            yaxis=dict(
                ticktext=[f'{y:.1f}' for y in np.linspace(0.5, 1.5, 11)],
                tickvals=np.linspace(0.5, 1.5, 11)
            )
        ),
        width=1000,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    # 保存为交互式HTML文件
    fig.write_html(f'{file_path}/3D_Reward_Surface_S{seed}.html')
    print(f"交互式3D曲面图已保存: 3D_Reward_Surface_S{seed}.html")
    