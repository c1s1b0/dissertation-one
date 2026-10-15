"""局部两几何关系多父变量诊断入口。"""
import argparse
import json
from pathlib import Path
from paper1_active_req.experiment.reviewer_parent_ablation import run_parent_ablation

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='两几何关系局部多父变量诊断')
    parser.add_argument('--output-dir',type=Path,default=Path('research_artifacts/reviewer_supplement_20261009/diagnostics/parent_ablation'))
    args=parser.parse_args()
    result=run_parent_ablation(Path(__file__).resolve().parents[1],args.output_dir.resolve())
    print(json.dumps({'prediction_count':result['prediction_count'],'comparisons':result['comparisons']},ensure_ascii=False,indent=2))
