"""仅对完整且绑定一致的审稿后补充实验生成规格级统计和图表。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from paper1_active_req.evaluation.v3_statistics import aggregate_primary_by_spec, holm_adjust_p_values
from paper1_active_req.experiment.reviewer_runner import (
    SEEDS, POLICIES, LAMBDAS, job_plan, json_hash, protected_inputs, protocol_settings, sha256,
)
from paper1_active_req.generator.v3_relational_specs import V3RequirementTask

PAIRED_METRICS = ['dcrr_at_8', 'rcrr_at_8', 'dcrr_auc', 'rcrr_auc', 'recovery_increment', 'answer_cost']
REQUIRED_METRICS = [f'{prefix}_at_{budget}' for prefix in ['dcrr', 'rcrr'] for budget in range(9)] + PAIRED_METRICS[2:] + ['actual_question_count']
PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / 'research_artifacts/reviewer_supplement_20261009'


def validate_records(records: list[dict], expected: dict[str, dict], protocol_hash: str,
                     outcome: dict, records_hash: str) -> None:
    """拒绝未完成、缺失、重复、失败、协议不符或键对应元数据被改动的记录。"""
    if outcome.get('status') != 'complete':
        raise ValueError('execution outcome must be complete')
    if outcome.get('protocol_hash') != protocol_hash or outcome.get('records_hash') != records_hash:
        raise ValueError('outcome protocol/records hash mismatch')
    keys = [row.get('key') for row in records]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate record key')
    if set(keys) != set(expected) or outcome.get('record_count') != len(records) or outcome.get('expected_count') != len(expected):
        raise ValueError('record key/count coverage is incomplete')
    for row in records:
        if row.get('status') != 'success':
            raise ValueError('failed record retained; complete statistics prohibited')
        if row.get('protocol_hash') != protocol_hash:
            raise ValueError('record protocol hash mismatch')
        if any(row.get(key) != value for key, value in expected[row['key']].items()):
            raise ValueError(f'record metadata mismatch: {row["key"]}')


def _comparisons(specs: pd.DataFrame, seed: int, baselines: tuple[str, ...], n_specs: int,
                 family: str, by_lambda: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    subsets = specs.groupby('lambda_cost', sort=True) if by_lambda else [(0., specs)]
    for lam, subset in subsets:
        for metric in PAIRED_METRICS:
            table = subset.pivot(index='spec_id', columns='policy_name', values=metric)
            if len(table) != n_specs or table.isna().any().any():
                raise ValueError('paired comparison requires complete specification pairs')
            for baseline in baselines:
                difference = (table['tr_mla'] - table[baseline]).to_numpy(dtype=float)
                sampled = difference[rng.integers(n_specs, size=(10000, n_specs))].mean(axis=1)
                low, high = np.quantile(sampled, [.025, .975])
                if np.all(difference == 0):
                    stat, p = 0., 1.
                else:
                    test = wilcoxon(difference, alternative='two-sided')
                    stat, p = float(test.statistic), float(test.pvalue)
                rows.append(dict(seed=seed, method='tr_mla', baseline=baseline, metric=metric,
                                 lambda_cost=lam, n_specs=n_specs, mean_difference=float(difference.mean()),
                                 ci_low=float(low), ci_high=float(high), wilcoxon_statistic=stat,
                                 p_value=p, bootstrap_replicates=10000, bootstrap_seed=seed,
                                 analysis_label='审稿后探索性', holm_family=family))
    result = pd.DataFrame(rows)
    result['holm_p_value'] = holm_adjust_p_values(result.p_value.tolist())
    return result


def primary_statistics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """λ=0，m50/m70先在每规格内平均；每种子六指标乘三个基线共18项。"""
    source = frame[(frame.lambda_cost == 0) & (frame.group == 'multiseed')].copy()
    numeric = [col for col in source.select_dtypes(include='number') if col not in ['seed', 'missing_rate', 'lambda_cost']]
    all_specs, all_means, all_comparisons = [], [], []
    if set(source.seed) != set(SEEDS):
        raise ValueError('primary statistics require all three protocol seeds')
    for seed in SEEDS:
        subset = source[source.seed == seed]
        if set(subset.policy_name) != set(POLICIES):
            raise ValueError('primary policy roster mismatch')
        raw = subset[['spec_id', 'policy_name', 'missing_rate', *numeric]].copy()
        raw['crr_at_8'] = raw.rcrr_at_8
        specs = aggregate_primary_by_spec(raw).rename(columns={'method': 'policy_name'})
        if not (specs.groupby('policy_name').size() == 60).all() or len(specs) != 240:
            raise ValueError('primary statistics require 60 specifications per policy')
        specs['seed'] = seed
        all_specs.append(specs)
        means = specs.groupby('policy_name')[numeric].mean().reset_index()
        means['seed'], means['n_specs'] = seed, 60
        means['missing_rate_scope'] = 'm50_m70_mean'
        all_means.append(means)
        all_comparisons.append(_comparisons(specs, seed, tuple(p for p in POLICIES if p != 'tr_mla'),
                                            60, f'seed_{seed}_primary_18'))
    means = pd.concat(all_means, ignore_index=True)
    descriptive = means.groupby('policy_name')[PAIRED_METRICS + ['actual_question_count']].agg(['mean', 'min', 'max'])
    descriptive.columns = ['_'.join(pair) for pair in descriptive.columns]
    descriptive = descriptive.reset_index()
    descriptive['seed_count'], descriptive['analysis_label'] = 3, '三种子描述性均值与范围；无跨种子合并检验'
    return pd.concat(all_specs, ignore_index=True), means, pd.concat(all_comparisons, ignore_index=True), descriptive


def cost_statistics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """预定义18规格m70；λ=0复用multiseed，同网格不选最优点。"""
    policies = ['typed_relational_myopic', 'tr_mla']
    selected = frame[frame.cost_subset & (frame.missing_rate == .7) & frame.policy_name.isin(policies)].copy()
    groups = ['seed', 'spec_id', 'policy_name', 'lambda_cost']
    if selected.duplicated(groups).any():
        raise ValueError('duplicate cost grid cell')
    all_comparisons = []
    for seed in SEEDS:
        subset = selected[selected.seed == seed]
        for policy in policies:
            rows = subset[subset.policy_name == policy]
            if rows.spec_id.nunique() != 18 or len(rows) != 72 or not rows.groupby('spec_id').lambda_cost.agg(set).map(lambda values: values == set(LAMBDAS)).all():
                raise ValueError('cost grid requires 18 complete specifications and four fixed lambdas')
        if not subset[subset.lambda_cost == 0].group.eq('multiseed').all() or not subset[subset.lambda_cost != 0].group.eq('cost').all():
            raise ValueError('cost grid must reuse multiseed lambda zero')
        all_comparisons.append(_comparisons(subset, seed, ('typed_relational_myopic',), 18,
                                            f'seed_{seed}_cost_24', by_lambda=True))
    positive_cost = selected.answer_cost.where(selected.answer_cost > 0)
    selected['recovery_per_cost'] = selected.rcrr_at_8 / positive_cost
    selected['increment_per_cost'] = selected.recovery_increment / positive_cost
    metrics = PAIRED_METRICS + ['actual_question_count', 'recovery_per_cost', 'increment_per_cost']
    means = selected.groupby(['seed', 'policy_name', 'lambda_cost'])[metrics].mean().reset_index()
    means['n_specs'], means['missing_rate_scope'] = 18, 'm70_predefined_mechanism_first_6'
    return selected.sort_values(groups), means, pd.concat(all_comparisons, ignore_index=True)


def load_complete_records(root: Path, project: Path) -> tuple[pd.DataFrame, dict]:
    """先核验协议、输入、数据集、结果哈希及2484预定义键，再允许任何汇总写入。"""
    protocol = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
    outcome = json.loads((root / 'execution_outcome.json').read_text(encoding='utf-8'))
    binding = protocol['binding']
    digest = json_hash(binding)
    if digest != protocol.get('protocol_hash') or binding['settings'] != protocol_settings():
        raise ValueError('protocol binding/settings hash mismatch')
    actual_inputs = {key: sha256(path) for key, path in sorted(protected_inputs(project).items())}
    if actual_inputs != binding['input_hashes']:
        raise ValueError('protected input/source hash changed after protocol binding')
    expected = {}
    dataset_hashes = {}
    for seed in SEEDS:
        directory = root / 'datasets' / f'seed-{seed}'
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['protocol_hash'] != digest or manifest['seed'] != seed:
            raise ValueError('dataset protocol binding mismatch')
        for name, expected_hash in manifest['files'].items():
            actual_hash = sha256(directory / name)
            if actual_hash != expected_hash:
                raise ValueError('dataset file hash mismatch')
            dataset_hashes[f'seed-{seed}/{name}'] = actual_hash
        tasks = [V3RequirementTask.model_validate_json(line) for line in (directory / 'test_tasks.jsonl').read_text(encoding='utf-8').splitlines()]
        by_id = {task.task_id: task for task in tasks}
        if len(tasks) != 180 or len(by_id) != 180 or len({task.spec_id for task in tasks}) != 60:
            raise ValueError('test task roster mismatch')
        for job in job_plan(tasks, seed):
            task = by_id[job['task_id']]
            expected[job['key']] = dict(**job, spec_id=task.spec_id, mechanism=task.audit.mechanism,
                                       pivot=task.audit.pivot_field_id is not None, missing_rate=task.missing_rate)
    if len(expected) != 2484:
        raise ValueError('protocol expected job count must equal 2484')
    path = root / 'records/episodes.jsonl'
    raw = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    validate_records(raw, expected, digest, outcome, sha256(path))
    flattened = []
    for row in raw:
        metrics = row['metrics']
        if any(name not in metrics for name in REQUIRED_METRICS) or any(not np.isfinite(value) for value in metrics.values()):
            raise ValueError('missing or non-finite episode metrics')
        episode = row['episode']
        if episode['task_id'] != row['task_id'] or episode['policy_name'] != row['policy_name'] or len(episode['steps']) > 8:
            raise ValueError('episode binding/budget mismatch')
        if metrics['actual_question_count'] != len(episode['steps']):
            raise ValueError('question count differs from episode trace')
        metadata = {key: value for key, value in row.items() if key not in ['episode', 'metrics', 'engine_cache']}
        answers = [step['satisfiable_after_answer'] for step in episode['steps']]
        metadata['after_answer_all_sat'] = float(bool(answers) and all(value is True for value in answers))
        metadata['after_answer_unknown_count'] = sum(value is None for value in answers)
        flattened.append({**metadata, **metrics})
    return pd.DataFrame(flattened), dict(protocol_hash=digest, records_hash=sha256(path),
                                       protected_input_sha256=actual_inputs, dataset_sha256=dataset_hashes)


def _budget_table(source: pd.DataFrame, grouping: list[str]) -> pd.DataFrame:
    curve_columns = [f'{prefix}_at_{budget}' for prefix in ['dcrr', 'rcrr'] for budget in range(9)]
    mean = source.groupby(grouping)[curve_columns].mean().reset_index()
    counts = source.groupby(grouping).agg(n_tasks=('task_id', 'size'), n_specs=('spec_id', 'nunique')).reset_index()
    mean = mean.merge(counts, on=grouping)
    tidy = mean.melt(id_vars=grouping + ['n_tasks', 'n_specs'], var_name='metric_budget', value_name='mean')
    tidy['metric'] = tidy.metric_budget.str.split('_at_').str[0]
    tidy['budget'] = tidy.metric_budget.str.split('_at_').str[1].astype(int)
    return tidy.drop(columns='metric_budget')


def _plots(output: Path, budget: pd.DataFrame, costs: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'criticality_greedy': '#a26432', 'marginal_lookahead_no_relations': '#777777',
              'typed_relational_myopic': '#237a57', 'tr_mla': '#3659a6'}
    labels = {'criticality_greedy': 'Criticality-Greedy', 'marginal_lookahead_no_relations': 'No-relations lookahead',
              'typed_relational_myopic': 'Relational greedy', 'tr_mla': 'TR-MLA'}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True, constrained_layout=True)
    for col, seed in enumerate(SEEDS):
        for row, metric in enumerate(['dcrr', 'rcrr']):
            ax = axes[row, col]
            subset = budget[(budget.seed == seed) & (budget.metric == metric)]
            for policy in POLICIES:
                curve = subset[subset.policy_name == policy].sort_values('budget')
                ax.plot(curve.budget, curve['mean'], '-o', color=colors[policy], label=labels[policy], markersize=3)
            ax.set_title(f'Seed {seed}; n=60 specs')
            ax.set(xlabel='Question budget', ylabel=metric.upper(), ylim=(0, 1), xticks=range(9))
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8, loc='lower right')
    for extension in ['png', 'pdf']:
        fig.savefig(output / f'budget_dual_metrics.{extension}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for col, seed in enumerate(SEEDS):
        for policy in ['typed_relational_myopic', 'tr_mla']:
            points = costs[(costs.seed == seed) & (costs.policy_name == policy)].sort_values('lambda_cost')
            axes[0, col].plot(points.answer_cost, points.rcrr_at_8, '-', color=colors[policy], label=labels[policy])
            markers = {0.: 'o', .05: 's', .1: '^', .2: 'D'}
            for point in points.itertuples():
                axes[0, col].scatter(point.answer_cost, point.rcrr_at_8, marker=markers[point.lambda_cost], color=colors[policy], s=40)
                axes[1, col].scatter(point.lambda_cost, point.increment_per_cost, marker=markers[point.lambda_cost], color=colors[policy], s=40)
            axes[1, col].plot(points.lambda_cost, points.increment_per_cost, '-', color=colors[policy], label=labels[policy])
        axes[0, col].set(title=f'Seed {seed}; n=18 specs, m70', xlabel='Mean answer cost', ylabel='RCRR@8')
        axes[1, col].set(xlabel='Fixed cost penalty lambda', ylabel='Mean recovery increment / answer cost', xticks=LAMBDAS)
        for ax in axes[:, col]:
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    for extension in ['png', 'pdf']:
        fig.savefig(output / f'cost_grid.{extension}', dpi=180)
    plt.close(fig)


def summarize(root: Path = DEFAULT_ROOT, project: Path = PROJECT) -> Path:
    frame, manifest = load_complete_records(root, project)
    specs, means, comparisons, cross = primary_statistics(frame)
    details, cost_means, cost_comparisons = cost_statistics(frame)
    primary = frame[(frame.group == 'multiseed') & (frame.lambda_cost == 0)]
    output = root / 'summary'
    output.mkdir(parents=True, exist_ok=True)
    tables = dict(episode_metrics=frame, primary_spec_m50_m70=specs, primary_means=means,
                  primary_paired_comparisons=comparisons, cross_seed_descriptive=cross,
                  cost_episode_metrics=details, cost_means=cost_means, cost_paired_comparisons=cost_comparisons)
    all_task_means = primary.groupby(['seed', 'policy_name'])[PAIRED_METRICS + ['actual_question_count']].mean().reset_index()
    all_task_means['n_tasks'], all_task_means['n_specs'] = 180, 60
    all_task_means['missing_rate_scope'] = 'm30_m50_m70_task_mean_descriptive'
    tables['all_180_tasks_means_descriptive'] = all_task_means
    for grouping, name in [(['seed', 'policy_name', 'mechanism'], 'mechanism'),
                           (['seed', 'policy_name', 'missing_rate'], 'missingness'),
                           (['seed', 'policy_name', 'pivot'], 'pivot'),
                           (['seed', 'policy_name', 'mechanism', 'missing_rate', 'pivot'], 'stratified')]:
        metrics = primary.groupby(grouping)[PAIRED_METRICS + ['actual_question_count']].mean().reset_index()
        counts = primary.groupby(grouping).agg(n_tasks=('task_id', 'size'), n_specs=('spec_id', 'nunique')).reset_index()
        tables[f'{name}_means_descriptive'] = metrics.merge(counts, on=grouping)
        tables[f'{name}_budget_descriptive'] = _budget_table(primary, grouping)
    tables['primary_budget_m50_m70'] = _budget_table(primary[primary.missing_rate.isin([.5, .7])], ['seed', 'policy_name'])
    tables['cost_budget'] = _budget_table(details, ['seed', 'policy_name', 'lambda_cost'])
    cost_groups = ['seed', 'policy_name', 'lambda_cost', 'mechanism']
    cost_strata = details.groupby(cost_groups)[PAIRED_METRICS + ['actual_question_count', 'increment_per_cost']].mean().reset_index()
    cost_counts = details.groupby(cost_groups).agg(n_specs=('spec_id', 'nunique')).reset_index()
    tables['cost_mechanism_means_descriptive'] = cost_strata.merge(cost_counts, on=cost_groups)
    inference_counts = [column for column in primary if column.startswith('inference_') and column.endswith(('total', 'correct'))]
    inference = primary.groupby(['seed', 'policy_name'])[inference_counts].sum().reset_index()
    inference['pooled_inference_precision'] = inference.inference_exact_correct / inference.inference_total.where(inference.inference_total > 0)
    inference['pooled_tolerance_precision'] = inference.inference_tolerance_correct / inference.inference_total.where(inference.inference_total > 0)
    inference['n_tasks'], inference['n_specs'] = 180, 60
    tables['inference_counts_descriptive'] = inference
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False, encoding='utf-8-sig')
    _plots(output, tables['primary_budget_m50_m70'], cost_means)
    manifest.update(status='complete', analysis_label='审稿后探索性', record_count=len(frame),
                    summary_source_sha256=sha256(Path(__file__)), bootstrap_replicates=10000,
                    bootstrap_seed=list(SEEDS), primary_holm_family='每种子6指标×3基线=18项',
                    cost_holm_family='每种子6指标×4固定lambda=24项',
                    n_primary_specs_per_seed=60, n_all_tasks_per_seed_policy=180, n_cost_specs_per_seed=18,
                    cost_zero_reused=True, cross_seed_analysis='仅描述性均值和范围，不拼接确认性检验')
    manifest['output_sha256'] = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != 'summary_manifest.json'}
    (output / '统计说明.md').write_text(
        '# 补充实验统计说明\n\n本汇总仅在execution_outcome为complete、协议及所有保护输入哈希一致、'
        '2484条预定义键全部齐全且无失败后生成。2160条λ=0主记录与324条非零λ成本记录分别分析。'
        '\n\n主分析每种子60规格：先在每规格内平均m50/m70，再比较TR-MLA与三个基线。'
        '六指标为DCRR@8、RCRR@8、两AUC、恢复增量与回答成本。10000次bootstrap，固定种子；'
        '双侧Wilcoxon，种子内18项同族Holm修正。问题数为描述性。'
        '\n\nall_180_tasks_means_descriptive包含每种子每策略180任务（m30/m50/m70）平均，'
        '不等同于主分析n=60规格。机制、缺失率及桥接/控制结果与预算轨迹属于描述性分层。'
        '三种子之间只报均值及范围，不将原正式实验并入新确认性分析。'
        '\n\n成本实验每种子预定义18规格、m70，固定λ=0、0.05、0.10、0.20；λ=0复用主记录。'
        '所有点均报告，不选择最佳λ。成本配对为探索性，每种子24项同族Holm。单位成本指标先在'
        '单条轨迹计算再取平均；成本为零时置缺失。图中λ为固定网格；上下两排的圆、方、三角、菱形'
        '分别对应λ=0、0.05、0.10、0.20，重合点保留。'
        '\n\nDCRR只计直接观察且精确正确，RCRR包含共同训练模型可靠且精确正确推断，'
        '工程容差正确另作诊断；推断统计限预算8冻结关键集。after_answer_all_sat与unknown'
        '仅描述有回答轨迹的回答后检查，不冒充初始状态的完整可行性检验。'
        '\n\n全部结果为审稿后探索性，负差值保留。PNG/PDF使用英文图例便于投稿，中文字段解释在本说明。\n', encoding='utf-8')
    manifest['output_sha256']['统计说明.md'] = sha256(output / '统计说明.md')
    (output / 'summary_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='核验完整补充实验后输出统计及预算/成本图')
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    arguments = parser.parse_args()
    print(summarize(arguments.root))
