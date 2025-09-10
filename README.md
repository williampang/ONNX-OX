# 手写符号识别（浏览器本地 ONNX Runtime Web 推理）

本项目提供：
- 训练脚本（PyTorch）并导出 ONNX（opset 13）
- 浏览器端前端（Canvas 采集 + 预处理 + onnxruntime-web 推理）
- 合成数据集生成脚本（无需手收集样本也能快速得到可用模型）
- 支持识别三种符号：圈（O）、叉（X）、勾（√），不确定时显示问号（?）

## 快速开始

1) 可选：创建并激活虚拟环境
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) 安装依赖
```
pip install -r requirements.txt
```

3) 生成合成数据集（几秒钟完成，现已支持小尺寸变体）
```
python gen_synth_dataset.py --out dataset --train 30000 --val 6000 --img-size 28
```
新增功能：
- 大规模数据集生成（默认36,000个样本，比之前增加7.5倍）
- 小尺寸形状变体（30%的样本使用更小的形状尺寸和更细的线条）
- 增强的形状多样性和随机性

将会生成：
```
dataset/
  train/
    O/*.png
    X/*.png
    checkmark/*.png
  val/
    O/*.png
    X/*.png
    checkmark/*.png
```

4) 本地训练并导出 ONNX 模型（现已包含详细结果输出）
```
python train_oxnet.py --data dataset --epochs 8 --batch 128 --onnx model/model.onnx --results results
```
新增功能：
- 详细的训练结果记录和JSON格式输出
- 训练历史跟踪（损失值、准确率变化）
- 自动生成训练摘要报告
- 模型性能评估报告

输出文件：
- `model/model.onnx` - ONNX模型文件
- `results/training_results.json` - 详细训练数据（JSON格式）
- `results/training_summary.txt` - 训练摘要报告（文本格式）

5) 本地打开前端进行推理
- 直接用浏览器打开 `index.html`，或
- 启动本地静态服务器（更稳妥）：
```
npx serve .
```
访问 http://localhost:3000

6) 页面使用
- 在画布写 O 或 X，点“识别”查看结果和置信度
- 也可用“保存样本 O/X”采集你自己的数据，完善数据集后再训练

## GitHub Actions：训练并产出 ONNX 构件

仓库已包含工作流 `.github/workflows/train.yml`：
- 手动运行：在 Actions 选项卡选择 "Train and Build ONNX"，点击 Run workflow
- 或在相关文件变更推送时自动触发（requirements.txt、训练脚本等）
- 工作流步骤：安装依赖 -> 生成合成数据 -> 训练并导出 -> 上传 `model/model.onnx` 为构件（artifact）
- 运行完成后，可在该 Workflow Run 的 Artifacts 区域下载 `model-onnx`

下载后可将 `model.onnx` 放回仓库的 `model/` 目录，前端即可使用。

## 性能优化与增强功能

### 数据集增强
- **大规模数据生成**：支持生成30,000+训练样本和6,000+验证样本
- **小尺寸形状变体**：30%的样本包含小尺寸的圈、叉、勾，提高小目标识别能力
- **改进的随机化**：增强形状的位置、大小、角度和厚度变化
- **自适应边距**：小尺寸变体使用更大边距和更细线条，保持形状清晰度

### 训练结果输出
- **详细性能记录**：完整的训练历史和验证指标
- **JSON格式数据**：便于程序化分析的结构化结果
- **可读性报告**：人类友好的训练摘要文本
- **模型性能评估**：包含准确率、损失值和训练时间等关键指标

### 当前模型性能
使用增强数据集训练的模型达到：
- **验证准确率**：100% (36,000样本数据集)
- **训练准确率**：99.98%
- **训练时间**：约76秒 (CPU)
- **支持形状**：圈（O）、叉（X）、勾（√）

## 模型约定

- Input: name = "input", shape = [1,1,28,28], dtype = float32
- Output: name = "output", shape = [1,3]（logits），类别顺序为 ["圈","叉","勾"]
- 不确定阈值：置信度低于60%时显示问号（?）

## 兼容性与常见问题

- 本项目使用 onnxruntime-web（WASM 后端），默认纯 CPU，无需 GPU/WebGL。
- 若推理报 shape 或 name 不匹配，请确认训练导出的 input_names/output_names 与前端一致（input/output）。
- 离线环境请将 `https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js` 下载到本地并用相对路径引用。

## License

MIT © 2025 williampang