# FOC BLDC/PMSM 仿真实验室

这是一个基于FOC（磁场定向控制）的BLDC/PMSM电机仿真平台，用于研究和验证各种控制算法。

## 功能特性

- BLDC/PMSM电机仿真模型
- FOC控制算法实现
- 弱磁控制算法
- 电机自学习功能
- 抗干扰控制算法
- 实时可视化：母线电压、id、iq曲线
- 参数可调的用户界面

## 项目结构

```
FOC_RD_LAB/
├── src/
│   ├── motor_model.py      # 电机模型
│   ├── foc_control.py      # FOC控制算法
│   ├── flux_weakening.py   # 弱磁控制
│   ├── self_learning.py    # 电机自学习
│   ├── disturbance_rejection.py  # 抗干扰控制
│   ├── visualization.py    # 可视化模块
│   └── simulation.py       # 主仿真程序
├── config/
│   └── motor_params.json   # 电机参数配置
├── tests/
│   └── test_scenarios.py   # 测试场景
└── requirements.txt       # 依赖包
```

## 安装与运行

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行仿真：
```bash
python src/simulation.py
```

## 使用说明

通过GUI界面可以调整各种参数，观察不同控制算法下的电机响应特性。

## 开发计划

- [x] 基础项目结构
- [ ] 电机模型实现
- [ ] FOC控制算法
- [ ] 弱磁控制
- [ ] 电机自学习
- [ ] 抗干扰控制
- [ ] 可视化系统
- [ ] 用户界面
- [ ] 测试场景