"""全量成本网格：规格配对、完整重放复验、全部固定参数汇总。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from run_full_cost_20261013 import ROOT,OLD,PROJECT,POLICIES,SEEDS,LAMBDAS,bind,plan,load_old,validate_coverage
from paper1_active_req.experiment.reviewer_runner import sha256,json_hash,_init_worker,_CONTEXT
from paper1_active_req.experiment.v3_episode import V3EpisodeResult
from paper1_active_req.evaluation.reviewer_metrics import reviewer_metrics
from paper1_active_req.evaluation.v3_metrics import _matches_truth
from paper1_active_req.evaluation.v3_statistics import holm_adjust_p_values
from paper1_active_req.domain.models import FieldStatus
from paper1_active_req.solver.engine import ConstraintEngine

PAIRED=['dcrr_at_8','rcrr_at_8','dcrr_auc','rcrr_auc','recovery_increment','answer_cost']
METRICS=PAIRED+['actual_question_count']
SCOPES=['m30','m50','m70','m50_m70_mean']

def scope_specs(frame,scope):
    rates={'m30':[.3],'m50':[.5],'m70':[.7],'m50_m70_mean':[.5,.7]}[scope]
    subset=frame[frame.missing_rate.isin(rates)]
    keys=['seed','spec_id','policy_name','lambda_cost']
    grouped=subset.groupby(keys)
    if not grouped.missing_rate.agg(list).map(lambda r:sorted(r)==rates).all():
        raise ValueError('incomplete or duplicate within-specification scope')
    result=grouped[METRICS].mean().reset_index(); result['scope']=scope
    return result

def verify():
    digest=bind()
    path=ROOT/'records/episodes.jsonl'
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    validate_coverage(rows,plan(),digest)
    outcome=json.loads((ROOT/'execution_outcome.json').read_text(encoding='utf-8'))
    if outcome['status']!='complete' or outcome['records_hash']!=sha256(path): raise ValueError('outcome binding mismatch')
    old={r['key']:r for r in load_old()}
    _init_worker(str(OLD),str(PROJECT))
    schema=_CONTEXT['schema']; steps=values=reused=0
    for seed in SEEDS:
        train,tasks,model=_CONTEXT['data'][seed]
        engine=ConstraintEngine(schema)
        for row in [r for r in rows if r['seed']==seed]:
            if row['provenance']=='reused':
                original=old[row['key']]
                if row['original_record_hash']!=json_hash(original) or any(row[k]!=original[k] for k in ['episode','metrics','wall_seconds','engine_cache']):
                    raise ValueError('reused record differs from frozen original')
                reused+=1
            elif row['provenance']!='new': raise ValueError('unknown provenance')
            episode=V3EpisodeResult.model_validate(row['episode']); task=tasks[row['task_id']]
            if episode.policy_name!=row['policy_name'] or episode.task_id!=task.task_id or episode.initial_state!=task.initial_state or len(episode.steps)>8:
                raise ValueError('episode binding mismatch')
            state=task.initial_state.public_copy()
            if engine.check_satisfiability(state).satisfiable is not True: raise ValueError('initial not SAT')
            for index,step in enumerate(episode.steps):
                if step.step_index!=index or step.field_id not in state.queryable_field_ids() or not schema[step.field_id].queryable:
                    raise ValueError('invalid step')
                if step.answer.status!=FieldStatus.OBSERVED or not _matches_truth(step.answer.value,task.ground_truth[step.field_id],schema[step.field_id]):
                    raise ValueError('oracle truth mismatch')
                state=state.apply_answer(step.field_id,step.answer)
                if step.satisfiable_after_answer is not True or engine.check_satisfiability(state).satisfiable is not True: raise ValueError('successor not SAT')
                steps+=1
            if state!=episode.final_state: raise ValueError('replay differs')
            metrics=reviewer_metrics(episode,task,schema,model)
            if set(metrics)!=set(row['metrics']): raise ValueError('metric schema mismatch')
            for key,value in metrics.items():
                if not math.isfinite(value) or not math.isclose(value,row['metrics'][key],rel_tol=0,abs_tol=1e-12): raise ValueError('metric mismatch')
                values+=1
        print(json.dumps(dict(event='verified_seed',seed=seed)),flush=True)
    if reused!=1404: raise ValueError('reuse count mismatch')
    bind()
    payload=dict(status='passed',records=len(rows),reused=reused,new=len(rows)-reused,steps=steps,metric_values=values,
        records_hash=sha256(path),source_hash=sha256(Path(__file__)),protocol_hash=digest,
        checks=['full grid coverage','frozen sources and inputs','reused original episode and metrics equality',
        'schema and task and policy binding','step eligibility and sequence','oracle truth','initial and successor SAT','final-state replay','all original reviewer metrics'])
    (ROOT/'independent_verification.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    return pd.DataFrame([{**{k:v for k,v in row.items() if k not in ['episode','metrics','engine_cache']},**row['metrics']} for row in rows])

def comparisons(specs):
    rows=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        for scope in SCOPES:
            subset=specs[(specs.seed==seed)&(specs.scope==scope)]
            for kind in ['within','between']:
                contrasts=([(p,lam,p,0.) for p in POLICIES for lam in LAMBDAS[1:]] if kind=='within'
                    else [('tr_mla',lam,'typed_relational_myopic',lam) for lam in LAMBDAS])
                for policy,lam,base,base_lam in contrasts:
                    a=subset[(subset.policy_name==policy)&(subset.lambda_cost==lam)].set_index('spec_id')
                    b=subset[(subset.policy_name==base)&(subset.lambda_cost==base_lam)].set_index('spec_id')
                    if len(a)!=60 or len(b)!=60 or set(a.index)!=set(b.index): raise ValueError('requires 60 spec pairs')
                    for metric in PAIRED:
                        difference=(a[metric]-b[metric]).sort_index().to_numpy()
                        low,high=np.quantile(difference[rng.integers(60,size=(10000,60))].mean(axis=1),[.025,.975])
                        stat,p=(0.,1.) if np.all(difference==0) else wilcoxon(difference,alternative='two-sided')
                        rows.append(dict(seed=seed,scope=scope,contrast=kind,policy=policy,lambda_cost=lam,
                            baseline=base,baseline_lambda=base_lam,metric=metric,n_specs=60,mean_difference=difference.mean(),
                            ci_low=low,ci_high=high,wilcoxon_statistic=stat,p_value=p,
                            holm_family=f'seed_{seed}_{kind}',bootstrap_replicates=10000,bootstrap_seed=seed))
    result=pd.DataFrame(rows); result['holm_p_value']=np.nan
    for family,subset in result.groupby('holm_family'):
        result.loc[subset.index,'holm_p_value']=holm_adjust_p_values(subset.p_value.tolist())
    return result

def output(frame):
    out=ROOT/'summary'; out.mkdir(exist_ok=True)
    specs=pd.concat([scope_specs(frame,s) for s in SCOPES],ignore_index=True)
    means=specs.groupby(['seed','scope','policy_name','lambda_cost'])[METRICS].mean().reset_index()
    means['n_specs']=60
    tests=comparisons(specs)
    cross=means.groupby(['scope','policy_name','lambda_cost'])[METRICS].agg(['mean','min','max'])
    cross.columns=['_'.join(c) for c in cross.columns]; cross=cross.reset_index()
    for name,data in [('episode_metrics',frame),('spec_metrics',specs),('means',means),('paired_comparisons',tests),('cross_seed_descriptive',cross)]:
        data.to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'typed_relational_myopic':'Relational greedy','tr_mla':'TR-MLA'}
    fig,axes=plt.subplots(2,3,figsize=(10.5,6.5),constrained_layout=True)
    selected=means[means.scope=='m50_m70_mean']
    for col,seed in enumerate(SEEDS):
        for policy,color in zip(POLICIES,['#237a57','#3659a6']):
            data=selected[(selected.seed==seed)&(selected.policy_name==policy)].sort_values('lambda_cost')
            axes[0,col].plot(data.lambda_cost,data.answer_cost,'o-',label=labels[policy],color=color)
            axes[1,col].plot(data.lambda_cost,data.rcrr_at_8,'o-',label=labels[policy],color=color)
        axes[0,col].set_title(f'Seed {seed}; 60 specifications')
        axes[0,col].set_ylabel('Mean synthetic answer cost'); axes[1,col].set_ylabel('RCRR@8')
        for ax in axes[:,col]: ax.set_xlabel('Cost penalty lambda'); ax.set_xticks(LAMBDAS); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(out/'full_cost_grid.pdf'); fig.savefig(out/'full_cost_grid.png',dpi=160); plt.close(fig)
    selected.to_csv(out/'full_cost_grid_plot_data.csv',index=False,encoding='utf-8-sig')
    lines=[r'\begin{table}[t]',r'\centering',r'\caption{Full fixed cost grid (post-review exploratory). Values are the descriptive mean [minimum, maximum] of the three seed-level means; each seed has 60 specifications with $m=0.5,0.7$ averaged within specification. Cost is synthetic, not monetary.}',r'\label{tab:full-cost-grid}',r'\small',r'\begin{tabular}{llccc}',r'\hline',r'Method & $\lambda$ & DCRR@8 & RCRR@8 & Answer cost \\',r'\hline']
    for policy in POLICIES:
        for lam in LAMBDAS:
            d=cross[(cross.scope=='m50_m70_mean')&(cross.policy_name==policy)&(cross.lambda_cost==lam)].iloc[0]
            vals=[f'{d[m+"_mean"]:.3f} [{d[m+"_min"]:.3f}, {d[m+"_max"]:.3f}]' for m in ['dcrr_at_8','rcrr_at_8','answer_cost']]
            lines.append(f'{labels[policy]} & {lam:.2f} & '+' & '.join(vals)+r' \\')
    lines.extend([r'\hline',r'\end{tabular}',r'\end{table}'])
    (out/'full_cost_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'scientific_interpretation.tex').write_text(
        'The cost-penalty analysis was extended to the complete supplementary test set: '
        '60 specifications per seed, three missingness levels, three seeds, two relational policies, '
        'and four fixed penalties ($\\lambda=0,0.05,0.10,0.20$), giving 4,320 episodes. '
        'The extension reused 1,404 exactly matching frozen episodes and computed the remaining 2,916. '
        'All grid points are reported; no penalty was selected as optimal using these test results. '
        'The paired unit is the specification (60 pairs within each seed). Missingness levels are reported separately, '
        'with a further analysis averaging $m=0.5,0.7$ within each specification. '
        'Within-method contrasts subtract the $\\lambda=0$ result from each nonzero-penalty result; '
        'between-method contrasts subtract relational greedy from TR-MLA at the same penalty. '
        'Unadjusted 95\\% percentile paired-bootstrap intervals use 10,000 resamples. '
        'Two-sided Wilcoxon tests use Holm correction across 144 within-method tests per seed '
        '(four scopes, two methods, three penalties, six metrics), and separately across 96 between-method tests per seed. '
        'The confidence intervals are exploratory and are not simultaneous intervals. '
        'Seeds are summarized descriptively and are not pooled into a confirmatory sample. '
        'Answer cost is the accumulated schema-defined synthetic cost, not measured time or money.\n',encoding='utf-8')
    (out/'统计说明.md').write_text('# 全量成本网格统计说明\n\n完整4320轨迹全部验证后统计；无失败排除。每种子每个范围60规格；m50/m70先在规格内平均。'
        '\n\n全部λ与三个缺失率均列于means.csv；spec_metrics.csv保留配对单元。cross_seed_descriptive.csv仅报告三种子均值和范围。'
        '\n\npaired_comparisons.csv同时包含同方法非零λ减λ0，以及同λ TR-MLA减关系贪心。每种子同方法144项Holm同族，方法间96项另一个同族。'
        '六指标为DCRR@8、RCRR@8、两AUC、恢复增量、回答成本。每项10000次配对bootstrap，95%百分位区间未经多重修正；双侧Wilcoxon全零差值记p=1。'
        '\n\n所有分析均为审稿后探索性；不依据测试集选择最优λ，不把合成回答成本换算为金钱或真实时间。问题数仅描述性。\n',encoding='utf-8')
    manifest=dict(status='complete',records=4320,source_hash=sha256(Path(__file__)),protocol_hash=bind(),
                  output_sha256={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='summary_manifest.json'})
    (out/'summary_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(event='summary_complete',means=len(means),comparisons=len(tests))),flush=True)

if __name__=='__main__': output(verify())
