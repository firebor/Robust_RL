import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 生成数据点
x = np.linspace(-1, 1, 1000)
mean = 0
variance = 0.04
std = np.sqrt(variance)

# 计算高斯分布的概率密度函数
y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-(x - mean)**2 / (2 * variance))

# 创建图形
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', lw=2, label=f'高斯分布 (μ={mean}, σ²={variance})')

# 设置图形属性
plt.title('高斯分布图')
plt.xlabel('x')
plt.ylabel('概率密度')
plt.grid(True)
plt.legend()

# 保存图形
plt.savefig('gaussian_distribution.png')
plt.close() 