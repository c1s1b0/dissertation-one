# 组件候选保留实验：代码、参数与归档记录

本仓库提供角色级候选数量匹配评估的实验材料，包括冻结任务、组件目录、历史运行记录、离线分析、参数与种子、合成生成器和测试。复现读取历史模型输出记录，不向模型服务发送请求。

## 文件结构

| 路径 | 内容 |
|---|---|
| `experiment_parameters.json` | 实验规模、方法、种子、约束参数、统计与审计规则 |
| `requirements-offline.txt` | Python 依赖版本 |
| `src/paper2_dynamic_agents/experiment/` | 组件/任务生成器、数据结构、硬约束求解器与候选池重放 |
| `scripts/` | 归档重分析、Support 基线、精确召回、随机敏感性、结构实验及验证入口 |
| `tests/` | 12 项离线测试 |
| `research_artifacts/benchmarks/formal-20260816/` | 冻结组件目录与 A/B 各 200 个测试任务 |
| `research_artifacts/formal_runs_v2/`、`research_artifacts/formal_llm_runs/` | 原历史执行记录 |
| `research_artifacts/reconstruction-20261013/` | 主分析、精确召回、随机敏感性与结构实验结果 |
| `research_artifacts/cmc-offline-20261014/` | Support 逐次重放、16 项匹配统计、工作量和审计汇总 |
| `research_artifacts/cmc-audit-input-20261014/` | 请求审计白名单和来源哈希 |
| `research_artifacts/reviewer-supplement-20260820/` | 归档 agent/Omi pilot 记录与外部来源 |
| `文件校验清单.json` | 发布文件的 SHA-256 |

## 环境和运行

验证环境为 Python 3.12.7、numpy 1.26.4、pydantic 2.12.4。在仓库根目录执行：

```powershell
python -m pip install -r requirements-offline.txt
python scripts/verify_cmc_bundle_20261014.py
```

快速入口校验文件、运行两组共 12 项测试、重算 Support 和审计，并比较 5 份关键输出。完整重算：

```powershell
python scripts/verify_cmc_bundle_20261014.py --full
```

完整入口还重算历史任务分析、480 个结构实例、精确解集合召回及 10 个随机种子的敏感性，核对 11 份关键结果。两种验证入口在计算期间禁用 Python 网络连接；无需 API 密钥。

请在干净副本中复现。脚本会更新生成结果，计时字段依硬件变化；验证排除本机计时差异，但保留历史归档延迟。重复验证时需使用新的干净发布副本，因为文件校验清单对应原始发布字节。日志和核验报告由 `.gitignore` 排除。

逐项运行：

```powershell
python tests/test_reconstruction_20261013.py
python tests/test_cmc_offline_20261014.py
python scripts/reconstruct_paper2_20261013.py
python scripts/structural_stress_20261013.py
python scripts/reconstruction_diagnostics_20261013.py
python scripts/cmc_offline_20261014.py
```

正常使用 `cmc_offline_20261014.py` 的默认入口。`--export-audit` 是历史原始日志的导出接口；公开包使用已经提供的脱敏白名单输入。

## 生成器、参数与统计

组件生成器为 `catalog.py` 的 `build_catalog(20260816)`，产生 160 条记录；任务生成器为 `dataset.py` 的 `build_tasks(scene, 'test', 200, 20260816)`，场景为 A 和 B。结构生成器位于 `structural_stress_20261013.py`，各实例种子与参数见 `structural_manifest.json`。这些生成器均使用合成规则。

Support 保留原初始池，逐角色匹配归档 P 的候选配额，排序使用单变量可行性和公开二元兼容性，不读取完整解集合。每任务三次重复采用二元多数表决；任务配对自助法为 10,000 次、种子 20261013，显著性水平为 0.05。

当前匹配比较采用 `cmc-offline-20261014/matched_statistics_16.json` 的 16 项统一 Holm 校正。`reconstruction-20261013/matched_statistics.json` 是原 12 项比较的归档版本；历史消融为另一独立 16 项家族。

LLM 流程四组成功率为 1.00/0.90/0.92/0.90，数值规则为 1.00/0.90/0.91/0.85，Support 为 1.00/0.90/0.94/0.90。与两种规则控制的配对差异均未达到校正后的显著性标准。独立结构实验只测试非 LLM 选择器。

## 记录边界

历史服务路由为 `gpt-5.6-terra`，记录温度为零，底层模型身份未获独立核实。仓库只支持历史候选池和记录的离线重算，不能保证重新生成原始模型响应。

白名单审计含 1,840 条请求、39 条失败；30/1,200 个 task-repeat 出现内部重试或任务重跑，条件验证接受数为 1,801/1,809。token 记录只有 8 条正数、1,801 条零值、31 条缺失；170,229 为正数记录的部分和，不能代表完整 token 或费用。私密请求 ID、endpoint、原始响应和密钥不在本包中。

合成约束的可行性不能代替物理装配或工业验证；归档 agent/Omi pilot 也不能证明跨模型泛化。外部来源许可见 `THIRD_PARTY_NOTICES.md`，本包没有新增统一开源许可证。
