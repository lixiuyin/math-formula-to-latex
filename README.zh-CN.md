# 基于视觉 Transformer 的印刷体与手写体数学表达式识别与 LaTeX 转换

[English](README.md) | **中文**

> 一个端到端的 **ViT 编码器 + Transformer 解码器** 模型，将数学公式图像转换为 LaTeX
> 词元序列（PyTorch 实现）。以**完全相同**的架构与训练配方，在**印刷体**
> （`im2latex-100k`）与**手写体**（`MathWriting-human`）两类数据集上训练与评估。
> 本仓库同时包含完整的学位/课程论文（[`Document/main.pdf`](Document/main.pdf)，中文）。

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.1-ee4c2c)
![Tests](https://img.shields.io/badge/tests-25%20passing-brightgreen)

模型为轻量级纯 Transformer（**约 7.73M 参数**），将公式识别建模为图像描述任务：
图像 → ViT 图块特征 → 自回归解码器 → LaTeX 词元。无 CNN 主干、无语法规则，
公式结构完全由端到端学习获得。

---

## 主要结果

均在各数据集**独立测试划分**上评估。掩蔽损失/准确率在教师强制下逐词元计算；
BLEU-4 在**完全自回归**生成下计算（500 个测试样本，NLTK `method1` 平滑），
是反映真实推理质量的主报告指标。

| 数据集 | 词表 | 测试准确率（教师强制） | BLEU-4（贪婪） | BLEU-4（束搜索，k=4） |
|---|:---:|:---:|:---:|:---:|
| **im2latex-100k**（印刷体） | 504 | **90.14 %** | 0.6767 | **0.7012** |
| **MathWriting-human**（手写体） | 235 | 56.63 % | 0.0833 | 0.0786 |

- 在**印刷体**上，7.73M 参数、`50×200` 低分辨率输入即取得 **0.70 BLEU-4**，
  验证了纯 Transformer 架构在该任务上的有效性。
- **手写体**实验是一项严格控制变量的迁移研究（相同配方、相同 5 万训练样本）：
  大幅性能下降定量刻画了跨书写者泛化的难度，而非声称这是一个强手写识别器。

| 训练曲线（印刷体） | 预测 vs. 真实标签（印刷体） |
|---|---|
| ![训练曲线](Document/pic/training_history.png) | ![预测对照](Document/pic/predictions.png) |

| 训练曲线（手写体） | 预测 vs. 真实标签（手写体） |
|---|---|
| ![手写训练曲线](Document/pic/training_history_mathwriting.png) | ![手写预测对照](Document/pic/predictions_mathwriting.png) |

手写体曲线呈现更早、更大的过拟合（验证损失约在第 23 轮触底，训练损失仍持续下降），即上文分析所述。

| 印刷体示例（array / 行列式） | 手写体示例（积分式 ∫dV|Ψ|²=N，正确识别） |
|---|---|
| ![印刷体示例](Document/pic/demo_array_sample.png) | ![手写体示例](Document/pic/demo_mathwriting_sample.png) |

---

## 模型架构

<p align="center">
  <img src="assets/architecture.png" alt="模型架构：ViT 编码器 + Transformer 解码器" width="560">
</p>

| 项目 | 取值 |
|---|---|
| 输入 | `50 × 200` 灰度，`10 × 10` 图块 → 100 个图块 |
| 嵌入维度 | 256 |
| 编码器 | 8 层、4 头、Pre-LN、GELU MLP `[512, 256]` |
| 解码器 | 4 层、8 头、Post-LN、ReLU 前馈 `512 → 256` |
| 参数量 | **7.73M**（编码器 4.27M + 解码器 3.46M） |
| 最大序列长度 | 152（含 `[START]`/`[END]`） |

**训练配方。** AdamW（`β1=0.9, β2=0.98, ε=1e-9`，权重衰减 `1e-4`）配合 Transformer
原论文预热调度（`d_model⁻⁰·⁵·min(t⁻⁰·⁵, t·warmup⁻¹·⁵)`，预热 800 步、峰值学习率约
`2.2e-3`）、**全局范数梯度裁剪 1.0**、批大小 768、Dropout 0.1、掩蔽交叉熵（忽略填充）、
固定随机种子的 80/20 训练-验证划分、早停（patience 10，回退最优权重）。输出层叠加基于
训练标签分布拟合的**词频对数先验偏置**。

---

## 关键发现

- **梯度裁剪是必需项而非可选项。** 在批 768、预热峰值学习率附近，无裁剪的训练在第 7 轮
  发散并坍缩为只输出 `{`/`}`（准确率卡在约 16.4% 的边际频率）。裁剪 1.0 + 预热 800 后，
  两个数据集训练全程稳定。
- **教师强制准确率会系统性高估生成质量。** 90.14% 词元准确率对应仅 0.68–0.70 的 BLEU-4，
  差距源于自回归误差累积，故以 BLEU 作为真实场景指标。
- **束搜索只对校准良好的模型有益。** 印刷体上 +0.0245 BLEU，但在欠拟合的手写模型上**无增益**
  ——一个有用的反向对照。
- **印刷体 → 手写体的性能差距**可分解为：(1) 书写者分布偏移（验证 79% vs. 测试 56.63%）、
  (2) 样本量不足/过拟合、(3) 近正方形手写图被强拉成 `50×200` 的宽高比失真。

---

## 仓库结构

```
.
├── Code/                       # PyTorch 实现（vit_latex 程序包）
│   ├── main.py                 # 入口：训练 + 评估流水线
│   ├── vit_latex/
│   │   ├── config.py           # 全部超参数 + DATASETS 注册表
│   │   ├── data/               # LaTeX 分词器 + 图像预处理 / HF 加载
│   │   ├── models/             # ViT 编码器、解码器组件、完整模型 + 束搜索
│   │   ├── training/           # 掩蔽指标、预热调度器、训练循环
│   │   └── evaluation/         # 评估、BLEU、可视化、交互式演示
│   ├── tests/                  # pytest 测试（仅 CPU、快速）
│   └── outputs/                # 运行产物（图表、演示图）
└── Document/                   # 完整论文（XeLaTeX，中文）—— 见 main.pdf
    ├── main.pdf                # 预编译论文 PDF
    ├── body/                   # 原理 / 训练 / 模型 / 结果 各章
    └── pic/                    # 图表
```

代码层面的细节见 [`Code/README.md`](Code/README.md)。

---

## 快速开始

Python **3.11+**，PyTorch **≥2.1**（论文报告结果使用 PyTorch 2.12，单卡 NVIDIA RTX 5090，32 GB）。

```bash
cd Code
uv sync                      # 或：pip install -r requirements.txt
```

### 训练与评估

```bash
python main.py                                   # 在 im2latex（印刷体）上训练 + 评估
python main.py --dataset mathwriting --beam      # 手写体 + 贪婪/束搜索 BLEU 对比
python main.py --eval-only [--weights PATH]      # 仅评估已保存权重，跳过训练
```

首次运行会从 Hugging Face 下载数据集，检测到 GPU 时自动使用。非默认数据集的产物文件名带
`_<dataset>` 后缀，避免互相覆盖。

### 测试

```bash
pytest        # 25 个测试，仅 CPU
```

---

## 数据集

| | 来源（Hugging Face） | 类型 |
|---|---|---|
| `im2latex-100k` | [`yuntian-deng/im2latex-100k`](https://huggingface.co/datasets/yuntian-deng/im2latex-100k) | 印刷体，已分词公式 |
| `MathWriting-human` | [`deepcopy/MathWriting-human`](https://huggingface.co/datasets/deepcopy/MathWriting-human) | 手写体，原始 LaTeX（加载时分词） |

每个数据集使用 5 万训练样本（为公平对比而设上限）；测试集为各数据集自带的独立划分（2,000 样本）。

---

## 论文

[`Document/main.pdf`](Document/main.pdf) 为完整论文（中文），涵盖 Transformer 与 ViT 原理、
训练原理、模型构建，以及详尽的结果与误差分析。重新编译（XeLaTeX + biber，`minted` 需
`-shell-escape`）：

```bash
cd Document
latexmk -xelatex -shell-escape main.tex
```

---

## 参考文献

- Vaswani et al., 2017. *Attention Is All You Need*（预热调度 §5.3，束搜索 §6.1）。
- Dosovitskiy et al., 2021. *An Image is Worth 16×16 Words*（ViT）。
- Wu et al., 2016. *Google's Neural Machine Translation System*（GNMT 长度惩罚）。
- Loshchilov & Hutter, 2019. *Decoupled Weight Decay Regularization*（AdamW）。
- 数据集：Deng et al.（im2latex-100k）；Gervais et al., 2025（MathWriting）。

## 许可

本仓库源代码以 [MIT 许可证](LICENSE) 发布。训练所用的第三方数据集
（im2latex-100k、MathWriting-human）以及论文（`Document/`）中可能包含的第三方图片，
其权利归各自所有者。
