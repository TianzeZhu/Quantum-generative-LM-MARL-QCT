# Dyck 串行量子模型实验交付

本交付对应单头 A1F1E3 模型：先以学习率 0.003 训练 100 轮，再保持模型与 Adam 状态，以 0.001 训练至累计 300 轮。正式结果以验证 NLL 选出的 best 和最后一次常规测试为准。

## 三部分内容

| 目录 | 内容 |
| --- | --- |
| 01_source | 模型、非线性预训练、正式训练、常规评估、运行入口、数据验证和文档图表生成代码 |
| 02_data | 原始训练/验证/常规测试数据、实际预训练权重与日志、重建数值样本、正式训练日志、best/latest、最终常规测试原始指标 |
| 03_documents | 中文 Word 实验报告、PDF 阅读版、PNG/PDF 图像、逐轮 CSV 和配置统计 |

建议先阅读 03_documents/串行量子模型_Dyck实验报告.docx。

## 数据说明

训练集 210 条，验证集 49 条，常规测试集 49 条，均为原始 JSONL 与逐行对应的 .src/.tgt 文件。原始词表保留，加载器将特殊 token ID 转为模型约定。

预训练 artifact 来自本次实际运行目录，SHA256 与正式 checkpoint 一致。softmax_reconstructed_samples.npz 与 sigmoid_reconstructed_samples.npz 是按 JAX 0.7.1、原始种子和键分裂流程重建的数值数据，不是原运行逐批保存的现场采样文件。各有 train_inputs/train_targets（1600×16×8）及 validation_inputs/validation_targets（256×8）。重建验证损失与原始记录一致，详见 reconstruction_metadata.json。

原始训练与评估日志、权重不修改。模型源码为交付时本地快照，原运行未记录服务器 Git 提交号。source_records.json 记录来源与摘要，delivery_manifest.json 为完整交付文件校验清单。

## 环境与校验

服务器已有相同环境可直接复用。新建文件夹环境示例：

```bash
cd /path/to/project_delivery
conda env create --prefix /path/to/dyck_env --file 01_source/model/environment_h200.yml
/path/to/dyck_env/bin/python 01_source/verify_delivery.py
```

训练环境采用 Python 3.11.15、JAX/JAXlib 0.7.1、NumPy 2.2.6、PennyLane 0.44.1；CUDA 后端。图表与文档重建另需 Matplotlib、python-docx；仅阅读文档不需要这些依赖。

## 运行最终常规评估

以下操作新建输出目录，不覆盖归档结果。默认使用包内 best 和对应预训练 artifact。

```bash
ROOT=/path/to/project_delivery
PYTHON=/path/to/dyck_env/bin/python
EVAL_DIR=/path/to/new_dyck_evaluation
mkdir -p "$EVAL_DIR"
nohup env PYTHON="$PYTHON" EVAL_DIR="$EVAL_DIR" \
  bash "$ROOT/01_source/run_evaluation.sh" \
  > "$EVAL_DIR/evaluate.log" 2>&1 < /dev/null &
```

评估覆盖常规测试全部 49 条，执行 teacher forcing 指标和贪心生成，最大解码长度 6。结果写入 EVAL_DIR/training.jsonl。若需要评估新的重现实验，给 run_evaluation.sh 追加 --checkpoint /path/to/new_run/best。

## 重新执行两阶段训练

```bash
ROOT=/path/to/project_delivery
PYTHON=/path/to/dyck_env/bin/python
RUN_DIR=/path/to/new_dyck_training
mkdir -p "$RUN_DIR"
nohup env PYTHON="$PYTHON" RUN_DIR="$RUN_DIR" \
  bash "$ROOT/01_source/run_training.sh" \
  > "$RUN_DIR/train.log" 2>&1 < /dev/null &
```

脚本要求新目录，保持 0.003 → 0.001 两阶段方案，每轮验证、每 50 step 保存。模型数学实现不变；delivery_runner.py 只处理相对路径和交付数据指纹兼容：先验证交付文件逐字节摘要，再返回原实验数据标识，保留原 checkpoint 校验所需的标识连续性。它不声称重新计算了完整来源目录摘要。

原始目录中的直接入口保留用于源码审计；交付后实际运行请使用 delivery_runner.py 或上述 shell 入口。不要覆盖 02_data/training 中归档的模型和日志。

## 重建数值数据、图表与 Word

```bash
python 01_source/regenerate_pretraining.py
python 01_source/plot_delivery_figures.py
python 01_source/build_word_report.py
```

数值数据重建要求 JAX 0.7.1；图表重建需要 NumPy/Matplotlib；Word 重建需要 python-docx。重新生成文件后，原交付清单可能因文件容器元数据变化而不再匹配，应另存重建结果并重新制作清单。默认提供的 Word 和 PDF 已完成版面检查，无需先执行重建命令。
