import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

seeds=[42,123,0,2025,7,100,999,73,66]
noise=0.5
os.makedirs(f'./res/heatmap/PDSAC_n{noise}', exist_ok=True)

# 生成示例数据
data = np.random.rand(20, 20)  # 10x10的随机数据
print(data.shape)


# 创建热力图
plt.figure(figsize=(10, 8))
sns.heatmap(data, 
            annot=True,  # 显示数值
            fmt='.0f',   # 数值格式
            cmap='YlOrRd',  # 颜色映射
            square=True)  # 保持正方形

# 添加标题
plt.title('Mass')

# 保存图片
plt.savefig(f'./res/heatmap/PDSAC_n{noise}/Mass_{seed}.png', dpi=300, bbox_inches='tight')
plt.close()