# FOC仿真系统性能优化指南

## 概述

本指南提供了针对低性能电脑的FOC仿真系统优化建议，帮助您在有限的硬件资源下获得最佳的仿真体验。

## 轻量级版本

### 使用轻量级仿真程序

我们提供了专门为低性能电脑优化的轻量级版本：

```bash
python src/simulation_lite.py
```

轻量级版本的优化特性：
- 降低采样率（从0.0001s增加到0.0005s）
- 减少数据缓冲区大小（从1000减少到500）
- 简化可视化界面（从6个图表减少到4个）
- 可调节的更新间隔（默认0.1秒）
- 性能监控和FPS显示

### 配置文件优化

使用轻量级配置文件：

```json
{
  "sample_time": 0.0005,
  "visualization_update_interval": 0.1,
  "data_buffer_size": 500,
  "enable_animation": false,
  "enable_3d_plots": false
}
```

## 性能优化策略

### 1. 采样率优化

**建议**：根据研究需求调整采样率
- 基本研究：0.001s（1kHz）
- 标准研究：0.0005s（2kHz）
- 精确研究：0.0001s（10kHz）

**方法**：修改配置文件中的`sample_time`参数

### 2. 可视化优化

**减少图表数量**：
- 基础版本：仅显示电流和速度
- 标准版本：显示电流、速度、电压和转矩
- 完整版本：显示所有参数

**降低更新频率**：
- 快速更新：0.05秒间隔
- 标准更新：0.1秒间隔
- 慢速更新：0.2秒间隔

**简化图表样式**：
- 减少网格线透明度
- 使用较细的线条
- 降低图表分辨率（DPI）

### 3. 数据缓冲区优化

**缓冲区大小建议**：
- 最小：100点（适合短时观察）
- 标准：500点（平衡性能和功能）
- 最大：1000点（完整历史记录）

### 4. 仿真速度控制

**实时仿真**：速度倍率 = 1.0
**快速仿真**：速度倍率 > 1.0
**慢速仿真**：速度倍率 < 1.0

## 系统级优化

### 1. 操作系统优化

**Windows系统**：
- 关闭不必要的后台程序
- 设置电源计划为"高性能"
- 增加虚拟内存

**Linux系统**：
- 使用轻量级桌面环境
- 关闭视觉效果
- 优化系统服务

### 2. Python环境优化

**使用轻量级库**：
```bash
# 替换matplotlib为更轻量的替代方案
pip install matplotlib --upgrade
pip install numpy --upgrade
```

**优化Python设置**：
- 使用Python 3.8+（性能更好）
- 安装Intel MKL优化的NumPy
- 考虑使用PyPy解释器

### 3. 硬件优化

**内存优化**：
- 确保至少4GB可用内存
- 关闭内存密集型应用
- 使用内存清理工具

**CPU优化**：
- 关闭不必要的进程
- 确保CPU不被过热降频
- 考虑超频（如果支持）

## 代码级优化

### 1. 算法优化

**简化计算**：
```python
# 使用向量化操作
import numpy as np

# 慢速方法
for i in range(len(data)):
    result[i] = data[i] * 2

# 快速方法
result = np.array(data) * 2
```

**减少函数调用**：
```python
# 慢速方法
def calculate(x):
    return x * 2 + 1

result = [calculate(x) for x in data]

# 快速方法
result = [x * 2 + 1 for x in data]
```

### 2. 内存优化

**使用生成器**：
```python
# 慢速方法
data = [i for i in range(1000000)]

# 快速方法
data = (i for i in range(1000000))
```

**及时释放内存**：
```python
import gc

# 处理完大数据后
del large_data
gc.collect()
```

### 3. 可视化优化

**使用draw_idle()替代draw()**：
```python
# 慢速方法
canvas.draw()

# 快速方法
canvas.draw_idle()
```

**减少重绘频率**：
```python
# 只在必要时更新
if frame_count % update_interval == 0:
    update_plots()
```

## 性能监控

### 1. 内置性能监控

轻量级版本包含性能监控功能：
- FPS显示
- 缓冲区使用情况
- 更新频率统计

### 2. 系统监控

**使用系统监控工具**：
- Windows：任务管理器、性能监视器
- Linux：top、htop、iotop
- 跨平台：psutil库

**关键指标**：
- CPU使用率
- 内存使用量
- 磁盘I/O
- GPU使用率（如适用）

## 故障排除

### 常见性能问题

1. **程序启动缓慢**
   - 检查Python版本
   - 减少导入的库
   - 使用延迟导入

2. **仿真运行缓慢**
   - 降低采样率
   - 减少可视化更新频率
   - 关闭不必要的功能

3. **内存使用过高**
   - 减小缓冲区大小
   - 及时清理数据
   - 使用更高效的数据结构

4. **界面响应迟缓**
   - 使用多线程
   - 减少GUI更新频率
   - 简化界面元素

### 性能测试

**基准测试脚本**：
```python
import time
import psutil
import os

def benchmark_simulation():
    start_time = time.time()
    start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    # 运行仿真
    # ...
    
    end_time = time.time()
    end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    print(f"执行时间: {end_time - start_time:.2f}秒")
    print(f"内存使用: {end_memory - start_memory:.2f}MB")

if __name__ == "__main__":
    benchmark_simulation()
```

## 推荐配置

### 低配置电脑（<4GB内存，双核CPU）

```json
{
  "sample_time": 0.001,
  "visualization_update_interval": 0.2,
  "data_buffer_size": 200,
  "enable_animation": false,
  "enable_3d_plots": false
}
```

**使用命令**：
```bash
python src/simulation_lite.py
```

### 中等配置电脑（4-8GB内存，四核CPU）

```json
{
  "sample_time": 0.0005,
  "visualization_update_interval": 0.1,
  "data_buffer_size": 500,
  "enable_animation": false,
  "enable_3d_plots": false
}
```

**使用命令**：
```bash
python src/simulation_lite.py
```

### 高配置电脑（>8GB内存，多核CPU）

```json
{
  "sample_time": 0.0001,
  "visualization_update_interval": 0.05,
  "data_buffer_size": 1000,
  "enable_animation": true,
  "enable_3d_plots": true
}
```

**使用命令**：
```bash
python src/simulation.py
```

## 总结

通过以上优化策略，即使在低性能电脑上也能流畅运行FOC仿真系统。关键是要根据硬件配置合理调整参数，在功能和性能之间找到平衡点。

建议从轻量级版本开始，根据实际需求逐步启用更多功能。如果仍有性能问题，可以考虑升级硬件或使用更专业的仿真工具。