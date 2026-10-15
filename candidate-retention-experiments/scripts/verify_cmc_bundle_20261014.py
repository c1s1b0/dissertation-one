"""独立解包验证：先核验清单，禁网运行，比较非计时结果；只能在干净副本执行。"""
import argparse,contextlib,csv,hashlib,json,runpy,socket,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TIME_FIELDS={'selection_ms','solver_ms','elapsed_ms','mean_selection_ms','mean_solver_ms'}
def comparable(obj):
    if isinstance(obj,dict):return {k:comparable(v) for k,v in obj.items() if k not in TIME_FIELDS}
    if isinstance(obj,list):return [comparable(v) for v in obj]
    return obj
def read(path):
    if path.suffix=='.json':return json.loads(path.read_text(encoding='utf-8'))
    if path.suffix=='.jsonl':return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    if path.suffix=='.csv':return list(csv.DictReader(path.open(encoding='utf-8-sig')))
    raise ValueError(path)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true');args=parser.parse_args()
    manifest=json.loads((ROOT/'文件校验清单.json').read_text(encoding='utf-8'))
    for n,h in manifest.items():
        p=(ROOT/n).resolve();assert p.is_relative_to(ROOT.resolve()),n
        assert hashlib.sha256(p.read_bytes()).hexdigest()==h,n
    tracked=['research_artifacts/cmc-offline-20261014/'+n+'.json' for n in ('support_replay','matched_statistics_16','matched_summary','support_workload','audit_summary')]
    if args.full:tracked+=['research_artifacts/reconstruction-20261013/'+n for n in ('primary_statistics.json','matched_statistics.json','matched_replay.jsonl','structural_tasks.csv','candidate_recall.csv','random_seed_records.json')]
    before={n:comparable(read(ROOT/n)) for n in tracked}
    raw={n:h for n,h in manifest.items() if n.startswith('research_artifacts/') and '/reconstruction-' not in n and '/cmc-offline-' not in n}
    def blocked(*a,**kw):raise RuntimeError('网络连接被离线复现入口禁用')
    socket.socket.connect=blocked;socket.socket.connect_ex=blocked;socket.create_connection=blocked
    commands=['tests/test_reconstruction_20261013.py','tests/test_cmc_offline_20261014.py']
    if args.full:commands+=['scripts/'+n+'.py' for n in ('reconstruct_paper2_20261013','structural_stress_20261013','reconstruction_diagnostics_20261013')]
    commands+=['scripts/cmc_offline_20261014.py']
    for command in commands:
        print('RUN',command,flush=True);sys.argv=[str(ROOT/command)]
        with (ROOT/'verification_run.log').open('a',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            try:runpy.run_path(str(ROOT/command),run_name='__main__')
            except SystemExit as exc:
                if exc.code not in (None,0):raise
    for n,expected in before.items():assert comparable(read(ROOT/n))==expected,('result mismatch',n)
    for n,h in raw.items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,('input changed',n)
    report=dict(status='PASS',full=args.full,network_connections_blocked=True,verified_package_members=len(manifest),compared_result_files=tracked,immutable_input_files=len(raw))
    (ROOT/'独立复现核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('INDEPENDENT_REPLAY_PASS',len(tracked),'result files; network blocked')
if __name__=='__main__':main()
