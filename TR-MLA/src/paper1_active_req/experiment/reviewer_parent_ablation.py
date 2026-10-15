"""两个几何关系的局部父变量适配诊断；不更改冻结关系配置或策略结果。"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

from paper1_active_req.domain.models import FieldStatus, FieldValue, RequirementState
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.evaluation.statistics import holm_adjust
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.relations.config import load_v3_relation_config
from paper1_active_req.relations.model import TypedRelationalModel

SEEDS = (20261011, 20261013, 20261017)
PARENTS = {'available_length_mm': ('workpiece_length_mm', 'travel_distance_mm'),
           'available_width_mm': ('workpiece_width_mm', 'side_clearance_mm')}
BOOTSTRAP_SEED = 20261009
BOOTSTRAP_REPETITIONS = 10000


def paired_error_stats(single_errors, multiple_errors, seed):
    single, multiple = np.asarray(single_errors, dtype=float), np.asarray(multiple_errors, dtype=float)
    if single.shape != multiple.shape or single.ndim != 1 or len(single) == 0:
        raise ValueError('paired error arrays must be nonempty equal-length vectors')
    differences = single - multiple
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(differences), size=(BOOTSTRAP_REPETITIONS, len(differences)))
    low, high = np.quantile(differences[draws].mean(axis=1), (.025, .975))
    outcome = wilcoxon(differences, alternative='two-sided') if np.any(differences != 0) else None
    return {'n_specs': len(single), 'mean_absolute_error_improvement': float(np.mean(differences)),
            'bootstrap_ci95': [float(low),float(high)], 'bootstrap_repetitions': BOOTSTRAP_REPETITIONS,
            'bootstrap_seed': seed, 'wilcoxon_statistic': float(outcome.statistic) if outcome else 0.,
            'wilcoxon_p': float(outcome.pvalue) if outcome else 1.}


def _read(path):
    return [CompleteSpec.model_validate_json(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def _write_csv(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_parent_ablation(root: Path, output_dir: Path):
    schema_path, config_path = root/'configs/fields.yaml', root/'configs/v3_relations.yaml'
    schema = load_field_schema(schema_path)
    config = load_v3_relation_config(config_path)
    datasets = {seed: root/f'research_artifacts/reviewer_supplement_20261009/datasets/seed-{seed}' for seed in SEEDS}
    inputs = {'schema': schema_path, 'relations': config_path}
    for seed,path in datasets.items():
        inputs[f'{seed}_train'] = path/'train_complete_specs.jsonl'
        inputs[f'{seed}_test'] = path/'test_complete_specs.jsonl'
        inputs[f'{seed}_manifest'] = path/'manifest.json'
    protocol = {'analysis_label': 'post_review_exploratory_added_after_initial_diagnostics',
                'motivation': '原几何关系省略工程规则中的行程或侧间隙父变量，诊断表达不足',
                'seeds': list(SEEDS), 'train_n_per_seed': 180, 'test_n_per_seed': 60,
                'variants': {'single_parent': '冻结原配置', 'multiple_parent': {key:list(value) for key,value in PARENTS.items()}},
                'public_state': '两变体使用相同union父字段，target和其他字段置空',
                'fit': '原最小二乘及训练残差经验分位数，无校准或阈值优化',
                'confidence_threshold': config.confidence_threshold,
                'engineering_tolerances': {key:config.engineering_tolerances[key] for key in PARENTS},
                'prediction': '冻结可靠解析代表值：距95%区间中点最近的分位分支',
                'interval_quantiles': config.continuous_interval_quantiles,
                'bootstrap_repetitions': BOOTSTRAP_REPETITIONS, 'bootstrap_seed': BOOTSTRAP_SEED,
                'paired_statistic': '单父绝对误差减多父绝对误差，规格级配对均值；正值表示多父较低',
                'wilcoxon': '双侧，对60个规格级绝对误差差值检验',
                'multiplicity_family': '三seed×两target共六对统一Holm',
                'input_hashes': {key:{'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for key,path in inputs.items()},
                'source_hashes': {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (Path(__file__), root/'src/paper1_active_req/relations/model.py',
                                              root/'src/paper1_active_req/relations/continuous.py', root/'src/paper1_active_req/solver/rules.py')}}
    output_dir.mkdir(parents=True, exist_ok=True)
    # 必须先记录完整协议，才可进行任何模型fit；已有记录禁止覆盖。
    with (output_dir/'protocol.json').open('x', encoding='utf-8') as handle:
        json.dump(protocol,handle,ensure_ascii=False,indent=2)
    rows, summaries, comparisons, fitted = [], [], [], []
    modified_relations = tuple(relation.model_copy(update={'parent_field_ids': PARENTS[relation.target_field_id]})
                               if relation.target_field_id in PARENTS else relation for relation in config.relations)
    multiple_config = config.model_copy(update={'relations': modified_relations})
    for seed, dataset in datasets.items():
        train, test = _read(dataset/'train_complete_specs.jsonl'), _read(dataset/'test_complete_specs.jsonl')
        if len(train)!=180 or len(test)!=60:
            raise ValueError('predeclared train180/test60 counts required')
        models = {'single_parent': TypedRelationalModel.fit(train, schema, config),
                  'multiple_parent': TypedRelationalModel.fit(train, schema, multiple_config)}
        for target, parents in PARENTS.items():
            tolerance = config.engineering_tolerances[target]
            for name,model in models.items():
                factor = model.factors_by_target[target][0]
                coefficients = factor.global_fit.coefficients.tolist()
                fitted.append({'seed':seed,'target':target,'variant':name,
                               'parent_ids':list(factor.relation_spec.parent_field_ids), 'coefficients_intercept_then_parents':coefficients,
                               'residual_interval_quantiles':factor.global_fit.residual_interval_quantiles,
                               'residual_branch_quantiles':factor.global_fit.residual_quantiles,
                               'branch_probabilities':factor.global_fit.branch_probabilities})
                group = []
                for spec in test:
                    state = RequirementState(values={key:FieldValue(value=None,status=FieldStatus.HIDDEN,source='parent_ablation_mask')
                                                     for key in spec.values})
                    for key in parents:
                        state.values[key] = FieldValue(value=spec.values[key].value,status=FieldStatus.OBSERVED,source='public_union_parent')
                    posterior = model.posterior(state,target)
                    low,high = posterior.interval_low,posterior.interval_high
                    prediction = float(min(posterior.branches,key=lambda b:abs(float(b.value)-(low+high)/2)).value)
                    truth = float(spec.values[target].value)
                    row = {'seed':seed,'spec_id':spec.spec_id,'target':target,'variant':name,
                           'parent_count':len(factor.relation_spec.parent_field_ids), 'parameter_count':len(coefficients),
                           'public_state':state.model_dump_json(), 'truth':truth,'prediction':prediction,
                           'signed_error':prediction-truth,'absolute_error':abs(prediction-truth),'squared_error':(prediction-truth)**2,
                           'interval_low':low,'interval_high':high,'coverage95':low<=truth<=high,
                           'interval_width':high-low,'tolerance':tolerance,'width_over_tolerance':(high-low)/tolerance,
                           'confidence':posterior.confidence,
                           'reliable':posterior.confidence>=config.confidence_threshold or high-low<=tolerance,
                           'engineering_correct':abs(prediction-truth)<=tolerance,
                           'exact_correct':abs(prediction-truth)<=1e-9}
                    rows.append(row)
                    group.append(row)
                summaries.append({'seed':seed,'target':target,'variant':name,'n_specs':len(group),
                                  'parent_count':len(factor.relation_spec.parent_field_ids),'parameter_count':len(coefficients),
                                  'mae':float(np.mean([x['absolute_error'] for x in group])),
                                  'rmse':float(np.sqrt(np.mean([x['squared_error'] for x in group]))),
                                  **{key:float(np.mean([x[source] for x in group])) for key,source in
                                     (('coverage95','coverage95'),('mean_width_over_tolerance','width_over_tolerance'),
                                      ('reliable_rate','reliable'),('engineering_correct_rate','engineering_correct'),('exact_correct_rate','exact_correct'))}})
            single = {x['spec_id']:x['absolute_error'] for x in rows if (x['seed'],x['target'],x['variant'])==(seed,target,'single_parent')}
            multi = {x['spec_id']:x['absolute_error'] for x in rows if (x['seed'],x['target'],x['variant'])==(seed,target,'multiple_parent')}
            ids = sorted(single)
            comparisons.append({'seed':seed,'target':target,**paired_error_stats([single[key] for key in ids], [multi[key] for key in ids],BOOTSTRAP_SEED)})
    for comparison,p in zip(comparisons,holm_adjust([x['wilcoxon_p'] for x in comparisons]),strict=True):
        comparison['holm_p'] = p
    _write_csv(output_dir/'predictions.csv',rows)
    _write_csv(output_dir/'variant_summary.csv',summaries)
    _write_csv(output_dir/'paired_statistics.csv',[{**x,'bootstrap_ci95':json.dumps(x['bootstrap_ci95'])} for x in comparisons])
    (output_dir/'fitted_parameters.json').write_text(json.dumps(fitted,ensure_ascii=False,indent=2),encoding='utf-8')
    result = {'prediction_count':len(rows),'comparisons':comparisons,'variant_summary':summaries,
              'analysis_label':protocol['analysis_label'], 'protocol_sha256':hashlib.sha256((output_dir/'protocol.json').read_bytes()).hexdigest(),
              'limitations':['局部两几何关系父变量适配，不是架构重制或工业验证',
                             '未改冻结配置、策略或已有TR-MLA结果','MAE下降不等于满足工程容差或达到可靠解析门槛',
                             '这是追加的审稿后探索性诊断，不是原预注册实验','三seed结果分开呈现，不拼接为独立样本总体']}
    (output_dir/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result
