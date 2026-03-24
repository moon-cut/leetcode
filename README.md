# leetcode
力扣刷题心得

---

## RNA逆向折叠 HGCN Demo

这个仓库新增了一个最小可运行的示例：把RNA二级结构（dot-bracket）映射到RNA序列（A/U/C/G），并用双曲图神经网络（HGCN）进行建模。

## 你需要下载哪些文件？

如果你只想跑这个 demo，**只需要下载当前仓库**（本仓库）即可，不需要额外下载 HazyResearch 仓库。

你至少需要以下文件：

- `train_rna_hgcn.py`
- `requirements.txt`
- `rna_hgcn/` 整个目录

## 本地运行（Linux/macOS）

```bash
git clone <你的仓库地址>
cd leetcode
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python train_rna_hgcn.py --epochs 200
```

## 本地运行（Windows PowerShell）

```powershell
git clone <你的仓库地址>
cd leetcode
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
python train_rna_hgcn.py --epochs 200
```

## 是否还要下载 https://github.com/HazyResearch 里的代码？

### 不需要（默认）
当前实现已经把核心思想（双曲空间映射 + 图消息传递）独立实现好了，直接可运行。

### 可选（进阶）
如果你后续想做更严格的对齐/复现，再去看 HazyResearch 的实现细节（例如训练配置、数据处理和实验脚本）即可，但这不是跑当前 demo 的前置条件。

## 目录

- `docs/rna_hgcn_workflow.md`: 任务流程与方法说明
- `rna_hgcn/structure.py`: dot-bracket到图
- `rna_hgcn/hyperbolic_layers.py`: Poincaré球运算与HGCN层
- `rna_hgcn/model.py`: 模型定义
- `rna_hgcn/data.py`: 数据样本和图构建
- `train_rna_hgcn.py`: 训练入口

## 常见问题

1. **报错 `No module named torch`**
   - 说明没安装 PyTorch，请先执行 `pip install -r requirements.txt`。

2. **我想换成自己的RNA数据**
   - 把你的 `(structure, sequence)` 样本替换 `rna_hgcn/data.py` 里的 `tiny_demo_dataset()`。

3. **如何和真实逆向折叠流程对齐**
   - 建议增加：配对互补约束、结构回折一致性损失（调用 ViennaRNA）、beam search 解码。
