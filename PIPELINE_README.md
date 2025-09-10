# 完整流水线使用指南 (Complete Pipeline Usage Guide)

## 概述 (Overview)

此项目实现了完整的 O/X 手写识别流水线：**创建模拟数据，训练模型，最后输出结果**

This project implements a complete O/X handwriting recognition pipeline: **Create mock data, train model, and output results**

## 一键运行 (One-Click Execution)

```bash
# 快速体验（小数据集）
python run_pipeline.py --train-samples 200 --val-samples 40 --epochs 3

# 完整训练（推荐）
python run_pipeline.py --train-samples 4000 --val-samples 800 --epochs 8

# 自定义参数
python run_pipeline.py --train-samples 1000 --val-samples 200 --epochs 5 --batch-size 64 --learning-rate 0.001
```

## 流水线步骤 (Pipeline Steps)

### 1. 创建模拟数据 (Generate Mock Data)
- 自动生成 O 和 X 的合成图像
- 包含随机变化：位置、大小、旋转、噪声
- 输出为 28x28 像素的 PNG 图像

### 2. 训练模型 (Train Model)
- 使用 PyTorch 训练卷积神经网络
- 实时显示训练和验证准确率
- 自动保存最佳模型

### 3. 导出 ONNX (Export to ONNX)
- 转换为 ONNX 格式用于 Web 推理
- 兼容 ONNX Runtime Web

### 4. 输出结果 (Output Results)
- 生成详细的 JSON 结果报告
- 包含数据集统计、模型信息、执行时间
- 提供下一步操作指导

## 生成的文件 (Generated Files)

```
dataset/                 # 合成数据集
├── train/
│   ├── O/              # 训练用 O 样本
│   └── X/              # 训练用 X 样本
└── val/
    ├── O/              # 验证用 O 样本
    └── X/              # 验证用 X 样本

model/
└── model.onnx          # 训练好的 ONNX 模型

pipeline_results.json   # 详细结果报告
```

## Web 界面测试 (Web Interface Testing)

1. 启动本地服务器：
```bash
python -m http.server 8080
```

2. 打开浏览器访问：http://localhost:8080

3. 在画布上绘制 O 或 X，点击"识别"按钮

## 参数说明 (Parameters)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset-dir` | `dataset` | 数据集输出目录 |
| `--train-samples` | `4000` | 训练样本数量 |
| `--val-samples` | `800` | 验证样本数量 |
| `--epochs` | `8` | 训练轮数 |
| `--batch-size` | `128` | 批次大小 |
| `--learning-rate` | `0.001` | 学习率 |
| `--model-path` | `model/model.onnx` | 模型输出路径 |

## 结果示例 (Example Results)

训练完成后，您将看到类似以下输出：

```
🎉 流水线执行完成!
============================================================
⏱️  总执行时间: 6.05 秒
📁 生成的文件:
   ✅ dataset
   ✅ model/model.onnx
   ✅ pipeline_results.json

🌐 下一步:
   1. 打开 index.html 在浏览器中测试模型
   2. 在画布上绘制 O 或 X，点击'识别'按钮
   3. 查看识别结果和置信度
```

## 故障排除 (Troubleshooting)

1. **依赖安装问题**：
```bash
pip install -r requirements.txt
```

2. **权限问题**：
```bash
chmod +x run_pipeline.py
```

3. **浏览器模型加载失败**：
   - 确保 `model/model.onnx` 文件存在
   - 使用 HTTP 服务器而不是直接打开 HTML 文件

## 技术架构 (Technical Architecture)

- **数据生成**: PIL (Python Imaging Library)
- **深度学习**: PyTorch
- **模型格式**: ONNX (Open Neural Network Exchange)
- **Web 推理**: ONNX Runtime Web
- **前端**: HTML5 Canvas + JavaScript

## 许可证 (License)

MIT License - 详见 LICENSE 文件