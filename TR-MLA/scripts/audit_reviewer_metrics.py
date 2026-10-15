"""只读重放两个历史种子，输出审稿后探索性双指标审计。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.evaluation.reviewer_metrics import reviewer_metrics
from paper1_active_req.evaluation.v3_statistics import aggregate_primary_by_spec, holm_adjust_p_values
from paper1_active_req.experiment.v3_runner import V3ExperimentRecord
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.generator.v3_relational_specs import V3RequirementTask
from paper1_active_req.relations.config import load_v3_relation_config
from paper1_active_req.relations.model import TypedRelationalModel


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'research_artifacts/reviewer_supplement_20261009/historical_audit'
METRICS = ['dcrr_at_8', 'rcrr_at_8', 'dcrr_auc', 'rcrr_auc', 'recovery_increment', 'answer_cost']


def paired_summary(frame: pd.DataFrame, seed: int, metrics: list[str]) -> pd.DataFrame:
    """每规格先平均m50/m70；TR-MLA与全部历史基线配对，种子内同族Holm。"""
    aggregated = aggregate_primary_by_spec(frame)
    rows = []
    rng = np.random.default_rng(seed)
    for metric in metrics:
        table = aggregated.pivot(index='spec_id', columns='method', values=metric)
        if 'tr_mla' not in table or table.isna().any().any():
            raise ValueError('paired comparison requires all complete policy/spec pairs')
        for baseline in sorted(set(table.columns) - {'tr_mla'}):
            differences = (table.tr_mla - table[baseline]).to_numpy(dtype=float)
            if not len(differences):
                raise ValueError('paired comparison requires at least one spec')
            boot = differences[rng.integers(len(differences), size=(10000, len(differences)))].mean(axis=1)
            lower, upper = np.quantile(boot, [0.025, 0.975])
            if np.all(differences == 0):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(differences, alternative='two-sided')
                statistic, p_value = float(result.statistic), float(result.pvalue)
            rows.append(dict(seed=seed, method='tr_mla', baseline=baseline, metric=metric,
                             n_specs=len(differences), mean_difference=float(differences.mean()),
                             ci_low=float(lower), ci_high=float(upper), p_value=p_value,
                             wilcoxon_statistic=statistic, bootstrap_replicates=10000,
                             analysis_label='审稿后探索性', missing_rate_scope='m50_m70_mean'))
    result = pd.DataFrame(rows)
    if len(result):
        result['holm_p_value'] = holm_adjust_p_values(result.p_value.tolist())
        result['holm_family'] = f'seed_{seed}_all_metrics_all_baselines'
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    schema_path, config_path = ROOT / 'configs/fields.yaml', ROOT / 'configs/v3_relations.yaml'
    schema, config = load_field_schema(schema_path), load_v3_relation_config(config_path)
    source_paths = [schema_path, config_path, Path(__file__),
                    ROOT / 'src/paper1_active_req/evaluation/reviewer_metrics.py',
                    ROOT / 'src/paper1_active_req/evaluation/v3_metrics.py',
                    ROOT / 'src/paper1_active_req/evaluation/v3_protocol.py',
                    ROOT / 'src/paper1_active_req/relations/model.py']
    independent = ROOT / 'research_artifacts/v3_robustness_20260810/runs/seed-20260813'
    runs = [(20260806, ROOT / 'data/generated/v3/train_complete_specs.jsonl',
             ROOT / 'data/generated/v3/test_tasks.jsonl', ROOT / 'results/v3/formal/episodes.jsonl'),
            (20260813, independent / 'development/train_complete_specs.jsonl',
             independent / 'release/test_tasks.jsonl', independent / 'records/episodes.jsonl')]
    source_paths.extend(path for _, *paths in runs for path in paths)
    before = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    all_rows, exclusions = [], []
    for seed, training_path, tasks_path, records_path in runs:
        training = [CompleteSpec.model_validate_json(line) for line in training_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        tasks = [V3RequirementTask.model_validate_json(line) for line in tasks_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        task_by_id = {task.task_id: task for task in tasks}
        if len(task_by_id) != len(tasks):
            raise ValueError('duplicate task IDs')
        records = [V3ExperimentRecord.model_validate_json(line) for line in records_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len({(r.task_id, r.policy_name) for r in records}) != len(records):
            raise ValueError('duplicate episode keys')
        model = TypedRelationalModel.fit(training, schema, config, use_relations=True)
        seed_rows = []
        for index, record in enumerate(records, 1):
            task = task_by_id[record.task_id]
            try:
                metrics = reviewer_metrics(record.episode, task, schema, model)
            except ValueError as error:
                if 'positive total criticality' not in str(error):
                    raise
                exclusions.append(dict(seed=seed, task_id=record.task_id, policy_name=record.policy_name, reason=str(error)))
                continue
            row = dict(seed=seed, task_id=record.task_id, spec_id=record.spec_id,
                       policy_name=record.policy_name, missing_rate=record.missing_rate,
                       mechanism=record.mechanism, pivot=record.pivot, **metrics)
            row.update({f'raw_{key}': value for key, value in record.metrics.items()})
            row['crr_at_8'] = metrics['rcrr_at_8']  # 旧聚合器要求的别名。
            row['common_minus_raw_crr_at_8'] = metrics['rcrr_at_8'] - record.metrics['crr_at_8']
            seed_rows.append(row)
            if index % 300 == 0:
                print(f'历史种子{seed}：重放{index}/{len(records)}', flush=True)
        frame = pd.DataFrame(seed_rows)
        frame.to_csv(OUTPUT / f'seed_{seed}_episodes.csv', index=False, encoding='utf-8-sig')
        aggregated = aggregate_primary_by_spec(frame.drop(columns=['seed']))
        aggregated.to_csv(OUTPUT / f'seed_{seed}_spec_m50_m70.csv', index=False, encoding='utf-8-sig')
        aggregated.groupby('method').mean(numeric_only=True).to_csv(OUTPUT / f'seed_{seed}_primary_means.csv', encoding='utf-8-sig')
        frame.groupby(['mechanism', 'missing_rate', 'policy_name']).mean(numeric_only=True).to_csv(
            OUTPUT / f'seed_{seed}_mechanism_missingness_means.csv', encoding='utf-8-sig')
        paired_summary(frame.drop(columns=['seed']), seed, METRICS).to_csv(
            OUTPUT / f'seed_{seed}_paired_comparisons.csv', index=False, encoding='utf-8-sig')
        all_rows.extend(seed_rows)
    pd.DataFrame(all_rows).to_csv(OUTPUT / 'all_episodes.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(exclusions, columns=['seed', 'task_id', 'policy_name', 'reason']).to_csv(OUTPUT / 'exclusions.csv', index=False, encoding='utf-8-sig')
    after = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    if before != after:
        raise RuntimeError('audit input changed during execution')
    manifest = dict(analysis_label='审稿后探索性历史重评价', input_and_source_sha256=before,
                    inputs_unchanged=True, record_count=len(all_rows), exclusions=len(exclusions),
                    model_fit_split='train_only', common_model='每个种子同一TypedRelationalModel(use_relations=True)',
                    exact_abs_tolerance=1e-9, engineering_tolerances=config.engineering_tolerances,
                    inference_scope='预算8冻结关键集；排除直接正确字段',
                    pairing='规格内m50/m70平均，TR-MLA与全部9个历史基线；每种子6指标×9基线同族Holm',
                    bootstrap_replicates=10000, bootstrap_seed='对应历史种子，顺序固定',
                    old_metrics_policy='所有原metrics以raw_前缀原样保留，不覆盖历史工件')
    (OUTPUT / 'audit_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUTPUT / '审计说明.md').write_text(
        '# 历史双指标审计\n\n原主种子20260806与独立20260813的全部历史策略轨迹，只重评价，不运行新算法。'
        '\n\nDCRR仅计直接观察且与完整真值一致的字段；RCRR额外计共同训练模型的可靠且精确正确推断。'
        '冻结关键集和1e-9精确阈值沿用旧协议；工程容差正确率只列诊断。推断计数限预算8冻结关键集。'
        '\n\n逐行CSV保留raw_旧指标，新rcrr_来自所有策略统一的关系模型，因此无关系策略的新旧数值可不同。'
        'crr_at_8是新rcrr_at_8给旧聚合器的别名，不代表旧结果。'
        '\n\n每种子按规格平均m50/m70后配对；10000次bootstrap与双侧Wilcoxon，'
        '同一种子全部6指标×9基线共54项检验Holm修正。未将缺失副本当作独立样本。'
        '\n\n结果是审稿后探索性，不替换原确认性分析。工程容差是基准假设，不构成工业认证。'
        '\n\n哈希见audit_manifest.json，记录所有读入数据、配置、直接评价源码及前后不变核验。\n', encoding='utf-8')
    print(json.dumps({'record_count': len(all_rows), 'exclusions': len(exclusions), 'inputs_unchanged': before == after}, ensure_ascii=False))


if __name__ == '__main__':
    main()
