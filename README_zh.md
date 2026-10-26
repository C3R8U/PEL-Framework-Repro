# 个性化大学英语教学框架——复现工具包

本仓库是可上传至 GitHub 的**代码与研究方法复现材料**，配套研究的三模块技术框架：学习者画像、教学内容生成、学习路径优化，并包含课堂统计、六配置消融、生成题目质量评价与图4—图8的汇总图绘制。

**真实性边界：** 附件只包含论文说明及图表均值，没有学生级原始记录、专家打分表、完整授权公开数据或模型训练权重。本仓库不伪造这些数据，也不声称重新得到论文中的显著性、效应量和教师评价指标。代码运行产生的结果来自使用者提供的实际数据。

## 启动

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python -m pytest -q
python scripts/verify_model.py
python scripts/optimize_path.py
python scripts/plot_figures.py
```

生成模型需要 GPU、独立的模型依赖和约15GB权重：

```bash
python -m pip install -r requirements-model.txt
python scripts/verify_model.py --online
python scripts/generate_exercises.py --words practice reflect --cefr B1 --count 2
```

`configs/model.json` 固定官方 Qwen2.5-7B-Instruct 的完整提交 SHA；`configs/components.json` 固定 BERT 模型版本。`prompts/` 保存完整提示词；生成输出中保存模型修订标识、生成设置和提示文件 SHA256。生成题目仅经过结构检查，语法、CEFR 难度与答案有效性必须由独立教师评价。

## 输入的数据

参见 `data/schemas/SCHEMAS.md`。所有课堂数据应采用匿名身份编号并放到未跟踪的 `data/raw/` 中。公开仓库只保留原附表图4—图8的汇总均值，无标准差时**绝不生成误差线或统计显著性**。托福公开试题本身不提供考生响应时长，MOOCCube聚类标签也不等同于真实的学习风格标注。

## 实证统计限制

- 16周班级干预的统计独立单位是**8个班级**，而不是235名学生。四名教师的配对班级随机化最多只有16种分配；双侧精确检验的最小非零 p 值为0.125。
- 20次多重插补敏感性分析的实现位于 `scripts/analyze_itt.py`，采用有条件的随机缺失假设，不能视为完全多层联合插补。
- 六配置交叉实验必须真实记录每一位参与者的六种条件、顺序及任务版本，不能复用原16周的两组前后测均值。
- 原始学生级数据、音频、专家真实评分、伦理批文及第三方受限语料不得提交到公开 GitHub。
- 本仓库提供可运行工具和明确的数据格式，但尚不构成对研究结果的完整独立复现证明。

更多命令、输入要求、限制与源代码说明见 [英文说明](README.md)。
