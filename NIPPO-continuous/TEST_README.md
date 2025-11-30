# NIPPO算法鲁棒性测试说明

## 概述

本测试代码参照SCSAC的`main_test_multi_cross.py`实现，用于评估NIPPO（PPO-continuous）算法在不同质量和摩擦力扰动下的鲁棒性表现。

## 文件说明

- `main_test_multi_cross.py`: 主测试脚本，实现多进程并行测试
- `test_script.sh`: 批量测试脚本，可以对多个环境和种子进行测试
- `TEST_README.md`: 本说明文档

## 测试原理

测试通过在MuJoCo环境中改变机器人的质量和摩擦力参数，评估训练好的策略在这些扰动下的表现：

- **质量扰动**: 在0.5倍到1.5倍原始质量范围内变化（51个采样点）
- **摩擦力扰动**: 在0.5倍到1.5倍原始摩擦力范围内变化（51个采样点）
- **测试网格**: 51×51 = 2601个测试点
- **评估方式**: 每个测试点运行10次episode，取平均奖励

## 使用方法

### 1. 准备工作

确保已经训练好NIPPO模型并提取了最佳模型：

#### 步骤1：训练模型
确保模型已训练并保存在`train_result`目录中。

#### 步骤2：提取最佳模型
运行以下命令提取最佳模型：

```bash
# 提取单个环境的最佳模型
python get_best_model.py --env_name="Walker2d-v2" --exp_name="PPO-continuous"

# 或批量提取所有环境的最佳模型
python get_best_model.py --batch --exp_name="PPO-continuous"
```

最佳模型将保存在以下目录结构：

```
best_model/
└── [实验名称]/
    └── [环境名称]/
        └── [环境名称]_PPO_seed[种子]/
            └── agent_[步数]_[奖励值]

train_result/
└── [实验名称]/
    └── [环境名称]/
        └── normalization_stats/
            └── [环境名称]_PPO_seed[种子]/
                └── normalization_stats.npy
```

### 2. 单个测试

运行单个环境和种子的测试：

```bash
python main_test_multi_cross.py \
    --env_name="Walker2d-v2" \
    --exp_name="PPO-continuous" \
    --seed=42
```

参数说明：
- `--env_name`: 环境名称（支持Walker2d-v2, Hopper-v2, HalfCheetah-v2等）
- `--exp_name`: 实验名称，需要与训练时一致
- `--seed`: 随机种子，需要与训练时一致

### 3. 批量测试

使用测试脚本进行批量测试：

```bash
./test_script.sh
```

或者：

```bash
bash test_script.sh
```

可以在脚本中修改以下参数：
- `seeds`: 要测试的种子列表
- `exp_name`: 实验名称
- `env_names`: 要测试的环境列表

### 4. 自定义测试

如果需要自定义测试参数，可以修改`main_test_multi_cross.py`中的`DISTURBANCE_DICT`：

```python
DISTURBANCE_DICT = {
    "Walker2d-v2": {
        "mass_list": np.linspace(0.5, 1.5, 51),      # 质量范围
        "friction_list": np.linspace(0.5, 1.5, 51)   # 摩擦力范围
    }
}
```

## 测试结果

### 结果保存位置

```
test_result/
└── [实验名称]_cross/
    └── [环境名称]/
        ├── result_numpy/
        │   └── [环境名称]_seed[种子].npy    # 测试结果矩阵
        ├── test_logs_logs/
        │   └── [环境名称]_PPO_seed[种子].log # 测试日志
        └── test_tensorboard_logs/
            └── [环境名称]_PPO_seed[种子]/    # TensorBoard日志
```

### 结果格式

- `.npy`文件包含51×51的奖励矩阵
- 行对应质量扰动（0.5x到1.5x）
- 列对应摩擦力扰动（0.5x到1.5x）
- 矩阵值为平均episode奖励

### 结果分析

可以使用以下Python代码加载和分析结果：

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载结果
results = np.load('test_result/PPO-continuous_cross/Walker2d-v2/result_numpy/Walker2d-v2_seed42.npy')

# 创建热图
plt.figure(figsize=(10, 8))
plt.imshow(results, cmap='viridis', origin='lower')
plt.colorbar(label='Average Reward')
plt.xlabel('Friction Multiplier')
plt.ylabel('Mass Multiplier')
plt.title('Robustness Test Results')
plt.show()
```

## 性能特点

- **多进程并行**: 使用20个进程同时测试，显著提高效率
- **内存优化**: 预先加载模型参数，避免重复加载
- **错误处理**: 完善的异常处理机制，保证测试稳定性
- **日志记录**: 详细的测试日志，便于调试和监控
- **资源管理**: 及时清理环境和模型，避免内存泄漏

## 注意事项

1. **依赖要求**: 确保安装了所有必要的依赖包（torch, gym, numpy等）
2. **模型文件**: 测试前确保训练好的模型文件存在且路径正确
3. **系统资源**: 多进程测试需要较多CPU和内存资源
4. **测试时间**: 完整的51×51网格测试需要较长时间（通常几十分钟到几小时）
5. **GPU使用**: 测试过程使用CPU，确保有足够的CPU核心数

## 故障排除

### 常见问题

1. **模型文件未找到**
   - 检查实验名称和种子是否正确
   - 确认模型文件是否存在

2. **归一化统计文件缺失**
   - 检查训练过程是否正确保存了归一化统计
   - 缺失时会给出警告但不影响测试

3. **内存不足**
   - 减少进程数量（修改`Pool(processes=20)`中的数字）
   - 或者使用更少的测试点

4. **环境兼容性问题**
   - 确保gym和mujoco版本兼容
   - 检查环境名称是否正确

### 调试建议

- 首先用单个测试点验证代码正确性
- 检查训练时的模型保存路径
- 查看详细的日志文件定位问题 