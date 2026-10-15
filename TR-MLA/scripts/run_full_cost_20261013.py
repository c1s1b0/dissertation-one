"""冻结全量成本网格，精确复用旧记录，逐条checkpoint；不改原算法。"""
from pathlib import Path
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from paper1_active_req.experiment.reviewer_runner import (
    SEEDS, LAMBDAS, _init_worker, _run_job, sha256, json_hash, protected_inputs, read_checkpoint, write_once)
from paper1_active_req.generator.v3_relational_specs import V3RequirementTask

PROJECT = Path(__file__).resolve().parents[1]
OLD = PROJECT / 'research_artifacts/reviewer_supplement_20261009'
ROOT = PROJECT / 'research_artifacts/full_cost_20261013'
POLICIES = ('typed_relational_myopic', 'tr_mla')

def load_old():
    return [json.loads(line) for line in (OLD/'records/episodes.jsonl').read_text(encoding='utf-8').splitlines()]

def plan():
    jobs = []
    for seed in SEEDS:
        tasks = [V3RequirementTask.model_validate_json(line) for line in
                 (OLD/'datasets'/f'seed-{seed}'/'test_tasks.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(tasks) != 180 or len({t.spec_id for t in tasks}) != 60:
            raise ValueError('task roster mismatch')
        for task in tasks:
            for policy in POLICIES:
                for lam in LAMBDAS:
                    jobs.append(dict(key=f'{seed}|{task.task_id}|{policy}|{lam:.2f}',seed=seed,
                        task_id=task.task_id,policy_name=policy,lambda_cost=lam,group='full_cost',
                        spec_id=task.spec_id,missing_rate=task.missing_rate,mechanism=task.audit.mechanism,
                        pivot=task.audit.pivot_field_id is not None))
    return jobs

def bind():
    inputs = protected_inputs(PROJECT)
    inputs.update({'old_protocol':OLD/'protocol.json','old_records':OLD/'records/episodes.jsonl',
                   'old_outcome':OLD/'execution_outcome.json','full_runner':Path(__file__)})
    for path in sorted((OLD/'datasets').rglob('*')):
        if path.is_file(): inputs['dataset:'+path.relative_to(OLD).as_posix()] = path
    binding = dict(input_hashes={k:sha256(v) for k,v in sorted(inputs.items())}, settings=dict(
        seeds=list(SEEDS),policies=list(POLICIES),lambda_grid=list(LAMBDAS),missing_rates=[.3,.5,.7],
        n_specs_per_seed=60,budget=8,expected=4320,expected_reused=1404,expected_new=2916,
        analysis='post-review exploratory; all fixed grid cells; no optimal lambda selection',
        scopes=['m30','m50','m70','m50_m70_mean'], bootstrap_replicates=10000,
        bootstrap_seed='dataset seed; sequential deterministic scope/policy/lambda/metric order',
        within_method_holm_family='per seed: 4 scopes x 2 policies x 3 nonzero lambdas x 6 metrics = 144',
        between_method_holm_family='per seed: 4 scopes x 4 lambdas x 6 metrics = 96',
        paired_metrics=['dcrr_at_8','rcrr_at_8','dcrr_auc','rcrr_auc','recovery_increment','answer_cost'],
        tests='two-sided Wilcoxon; zero differences p=1; percentile paired bootstrap; no cross-seed pooling',
        failures='retained; incomplete or failed grid prohibits complete statistics',cost_units='synthetic schema answer cost, not money'))
    payload = dict(binding=binding,protocol_hash=json_hash(binding))
    path = ROOT/'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != payload: raise ValueError('frozen binding changed')
    else: write_once(path,payload)
    return payload['protocol_hash']

def validate_coverage(rows,jobs,digest):
    expected = {j['key']:j for j in jobs}
    if len(rows) != len(expected) or {r['key'] for r in rows} != set(expected):
        raise ValueError('missing or duplicate grid cell')
    for row in rows:
        if row['status'] != 'success' or row['protocol_hash'] != digest: raise ValueError('failed or unbound row')
        if any(row.get(k)!=v for k,v in expected[row['key']].items()): raise ValueError('metadata mismatch')

def run(workers=6):
    digest = bind()
    jobs = plan()
    old = load_old()
    old_protocol = json.loads((OLD/'protocol.json').read_text(encoding='utf-8'))['protocol_hash']
    old_outcome = json.loads((OLD/'execution_outcome.json').read_text(encoding='utf-8'))
    if old_outcome['records_hash'] != sha256(OLD/'records/episodes.jsonl') or len(old)!=2484:
        raise ValueError('old result hash/count mismatch')
    if any(r['status']!='success' or r['protocol_hash']!=old_protocol for r in old):
        raise ValueError('old result not complete')
    old_by_key = {r['key']:r for r in old}
    if len(old_by_key)!=2484: raise ValueError('old duplicate')
    path = ROOT/'records/episodes.jsonl'
    prior = read_checkpoint(path,digest)
    expected = {j['key'] for j in jobs}
    if not set(prior)<=expected or any(r['status']!='success' for r in prior.values()):
        raise ValueError('invalid prior checkpoint; retained for investigation')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8',newline='\n') as stream:
        for job in jobs:
            if job['key'] not in prior and job['key'] in old_by_key:
                original = old_by_key[job['key']]
                row = {**original,**job,'protocol_hash':digest,'provenance':'reused',
                       'original_record_hash':json_hash(original),'original_protocol_hash':old_protocol}
                stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
                prior[row['key']] = row
        stream.flush()
        pending = [j for j in jobs if j['key'] not in prior]
        print(json.dumps(dict(event='start',total=4320,completed=len(prior),pending=len(pending))),flush=True)
        with ProcessPoolExecutor(max_workers=workers,initializer=_init_worker,initargs=(str(OLD),str(PROJECT))) as pool:
            futures = {pool.submit(_run_job,j):j for j in pending}
            for future in as_completed(futures):
                row = {**future.result(),'protocol_hash':digest,'provenance':'new'}
                stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n'); stream.flush()
                prior[row['key']] = row
                if len(prior)%25==0: print(json.dumps(dict(event='checkpoint',completed=len(prior),total=4320)),flush=True)
    validate_coverage(list(prior.values()),jobs,digest)
    bind()
    outcome=dict(status='complete',protocol_hash=digest,record_count=len(prior),records_hash=sha256(path),
        reused=sum(r['provenance']=='reused' for r in prior.values()),new=sum(r['provenance']=='new' for r in prior.values()),
        failed=sum(r['status']!='success' for r in prior.values()))
    target=ROOT/'execution_outcome.json'
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8')) != outcome: raise ValueError('outcome changed')
    else: write_once(target,outcome)
    print(json.dumps(outcome),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if not 1<=args.workers<=6: parser.error('workers must be 1..6')
    run(args.workers)
