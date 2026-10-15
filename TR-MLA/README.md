# TR-MLA：机械设计需求主动获取

本仓库提供 Typed Relational Marginal Lookahead（TR-MLA）的研究代码、合成基准、实验轨迹与评估工具。研究对象是在有限查询预算下获取缺失的机械设计需求。

## 内容

| 路径 | 内容 |
| --- | --- |
| `src/paper1_active_req` | 策略、关系模型、约束求解、数据生成与指标 |
| `configs` | 字段、成本、关系与实验配置 |
| `data/generated/v3`、`results/v3` | 原冻结实验的数据与结果 |
| `research_artifacts/reviewer_supplement_20261009` | 三种子补充实验、连续推断、冲突噪声、运行时间等诊断 |
| `research_artifacts/full_cost_20261013` | 完整成本网格：4,320条轨迹，含1,404条复用与2,916条新增 |
| `research_artifacts/public_engineering_20261013` | 公开工程需求来源、字段映射、360条回放轨迹与1,800个预算前缀 |
| `scripts`、`tests` | 运行、统计、验证脚本及测试 |
| `SHA256SUMS.json` | 本次发布文件完整性校验清单 |

## 环境

原实验环境为 Windows 11、Python 3.12.7。建议使用 Python 3.12，在仓库根目录建立独立虚拟环境。以下命令中的 `python` 应指向已激活的虚拟环境：

```shell
python -m venv .venv
# Windows PowerShell: ./.venv/Scripts/Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r research_artifacts/reviewer_supplement_20261009/复现依赖版本.txt
python -m pip install --no-deps -e .
```

精确依赖来自原实验环境。本次发布检查使用现有依赖进行迁移验证，不代表已在全新操作系统上验证安装。

## 核验已有结果

```shell
python -X utf8 scripts/verify_release_files.py
python -X utf8 scripts/verify_reviewer_portable.py
python -X utf8 -m pytest tests -k reviewer -p no:cacheprovider
python -X utf8 -m pytest research_artifacts/public_engineering_20261013/tests tests/test_full_cost_20261013.py -q -p no:cacheprovider
```

便携核验入口将历史清单中的绝对源码路径映射到当前仓库的相对位置，再检查原文件哈希、协议绑定、答案真值、约束和指标，不修改冻结记录。部分旧测试依赖未收录的历史稿件工程，故上面给出本包适用的测试范围。

## 重新运行与结果保护

`scripts/run_reviewer_supplement.py`、`scripts/run_full_cost_20261013.py` 和 `scripts/run_public_engineering_20261013.py` 是对应冻结实验的运行入口；统计入口为 `summarize_reviewer_supplement.py` 和 `summarize_full_cost_20261013.py`。这些脚本保留当时的输入绑定、输出目录和覆盖策略。请勿把原结果目录当作任意新实验的输出位置。新实验应在独立副本中明确建立新协议和新输出目录。

历史协议含原工作站路径，保留是为了维持原始哈希链；除上列便携核验外，不宣称全部旧运行器在任意目录均可直接重跑。算法源码和已保存结果没有为打包而修改。发布版便携核验包装器增加了只读比较：重复运行时对比现有验证记录，任何字段不一致即失败，不再尝试覆盖该文件。

## 指标与解释范围

- DCRR仅计正确的直接答案；RCRR额外计共同关系评估器接受的可靠且精确正确推断。初始RCRR可非零。历史CRR与修订后的指标应区分解释。
- 三个补充种子为20261011、20261013、20261017；以规格为统计单位，不能把重复掩码、预算前缀或复用记录当作独立项目。
- 完整成本网格覆盖四个成本系数，全部结果均保留；更早的18规格试验是先导诊断，不代表完整网格。主比较与完整网格共5,400条去重轨迹。
- 公开案例来自一个项目的两个子系统，只有四个可审计标量字段，不是设备实测或独立工业验证。来源快照及引用信息在 `sources` 和 `protocol.json`；没有可用的完整父字段—目标关系链，不能据此宣称真实场景下的关系推断优势。
- 结果没有证明TR-MLA普遍优于关系贪心；连续目标未通过原可靠推断门槛。成本为基准设定单位，不是测得的访谈时间或货币成本。
- 自然语言解析不属于本实验的执行管线；噪声回放不等同于真实LLM解析实验。

## 使用权限

本包仅包含实验材料，不包含论文正文、TeX稿件、审稿回复或图像附件。统计汇总CSV、原始记录和数值作图脚本保留，可按需重新生成数值图表。历史结果清单可能记载未随包分发的派生图表；当前包的文件完整性以根目录 `SHA256SUMS.json` 为准。

本包未替作者指定开源许可证。公开可访问不自动授予任意再分发或修改许可；第三方来源资料的权利归原权利人。正式发表后可补充论文DOI、引用格式和作者选定的许可证。
