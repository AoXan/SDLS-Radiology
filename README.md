## 🚀 Quick Start

### 1. Prerequisites

**Bash**

```
pip install torch transformers pandas numpy scikit-learn tqdm pillow
# Optional for full metrics:
pip install radgraph bert_score spacy
```

### 2. Pipeline Reproduction

#### Stage I: Data Preparation

Mine contrastive pairs (reports with and without history) and link images.

**Bash**

```
python data/mine_contrastive_pairs.py --clean_train_csv path/to/clean.csv --original_json path/to/mimic.json
python data/link_images.py --input_csv data/contrastive_pairs.csv --annotation_json path/to/mimic.json
```

#### Stage II: Vector Construction

**Step A: Extract Multi-layer Contextual Vectors (MCV)**

**Bash**

```
python core/extract_states.py --input_csv data/multimodal_pairs.csv --image_root /path/to/mimic-cxr
```

**Step B: Decompose & Compute Steering Vector**

* **For SDIV (Proposed):**
  **Bash**

  ```
  python core/decomposition.py --input_pkl data/hidden_states.pkl --method qr --output_path core/sdiv_vector.npy
  ```
* **For Global ICV (Baseline):**
  **Bash**

  ```
  python core/decomposition.py --input_pkl data/hidden_states.pkl --method pca
  ```

#### Stage III: Inference (Steering)

Apply the vector using the **SteerFair** strategy on Attention layers (Best performance in Table V).

**Bash**

```
python experiments/inference.py \
    --backend biomedgpt \
    --model_path /path/to/biomedgpt \
    --image_root /path/to/images \
    --input_csv data/multimodal_pairs.csv \
    --use_sdiv --icv_path core/sdiv_vector.npy \
    --strategy SteerFair --submodule attention \
    --strength -2.0
```

### 3. Evaluation

Calculate Hallucination Span Rate (HSR) and Clinical Metrics.

**Bash**

```
# Calculate HSR (Token-level)
python metrics/hsr.py --input_csv results/results.csv --output_csv results/metrics_hsr.csv

# Calculate Clinical Metrics (RadGraph, BERTScore, Hallucination Sentence Count)
python metrics/clinical.py --input_csv results/metrics_hsr.csv --output_csv results/final_metrics.csv
```

## 📊 Citation

If you use this code, please cite our TPAMI paper:

**Code snippet**

```
@article{li2026sdls,
  title={Suppressing Prior-Comparison Hallucinations in Radiology Report Generation via Semantically Decoupled Latent Steering},
  author={Li, Ao and others},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026}
}
```

```

---

### 第二部分：如何将本地文件夹完整替换 GitHub 仓库

这是“核弹级”操作。我们将不理会 GitHub 上原有的任何历史记录，直接把本地这个完美的版本强制推送到远端。

**前提条件**：
1.  你已经在 GitHub 上有一个仓库（假设地址是 `https://github.com/YourName/SDLS-Radiology.git`）。
2.  你的本地文件夹包含了所有重构后的代码，并且已经加上了上面的 `README.md`。

#### 步骤 1：创建 .gitignore (非常重要！)
防止把巨大的模型文件 (`.bin`, `.pth`) 或数据文件 (`.pkl`, `.npy`) 上传上去。
在项目根目录下创建一个名为 `.gitignore` 的文件，写入以下内容：

```text
# .gitignore
__pycache__/
*.pyc
*.pkl
*.npy
*.csv
.DS_Store
/models/
/results/
/data/MIMIC-CXR/
```

#### 步骤 2：初始化 Git 并“重新做人”

打开终端（Terminal），进入你的项目文件夹，执行以下命令：

1. **删除旧的 git 关联** (如果本地以前初始化过 git，先删掉，保证清白)：
   **Bash**

   ```
   rm -rf .git
   ```

   *(注意：Windows 用户如果是 PowerShell，用 `rd /s /q .git`，或者直接手动删除隐藏的 .git 文件夹)*
2. **重新初始化**：
   **Bash**

   ```
   git init
   ```
3. **添加所有文件**：
   **Bash**

   ```
   git add .
   ```
4. **提交代码**：
   **Bash**

   ```
   git commit -m "Refactor: Complete architectural overhaul aligning with TPAMI paper"
   ```
5. **切换分支名** (现在 GitHub 默认是 main)：
   **Bash**

   ```
   git branch -M main
   ```

#### 步骤 3：强制推送到远程 (覆盖一切)

1. **关联远程仓库** (替换为你自己的 GitHub 地址)：
   **Bash**

   ```
   git remote add origin https://github.com/YourName/SDLS-Radiology.git
   ```
2. **强制推送** (The Nuclear Option)：
   **Bash**

   ```
   git push -u origin main --force
   ```

**执行完这一步后，你的 GitHub 仓库将与你本地的完美重构版本一模一样。旧的杂乱代码和提交历史将全部消失。**

---

### 🎉 恭喜！

你现在拥有了一个：

1. **逻辑严密**（通过资深审计）。
2. **架构清晰**（Data/Core/Exp/Metrics 分离）。
3. **文档齐全**（包含 README）。
4. **线上同步**（GitHub 已更新）的开源级项目。

如果有任何特定的模块运行报错，或需要补充其他辅助脚本，随时告诉我！

```

```
