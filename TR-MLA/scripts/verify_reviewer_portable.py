"""移动目录后按协议相对键解析原文件，保留所有哈希再运行独立核验。"""
from pathlib import Path
import json
import runpy
import sys


def main():
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / 'src'))
    sys.path.insert(0, str(project / 'scripts'))
    from paper1_active_req.experiment import reviewer_runner
    import summarize_reviewer_supplement

    def relocated_inputs(location):
        location = Path(location)
        inputs = {
            'formal_episodes': location / 'results/v3/formal/episodes.jsonl',
            'formal_freeze': location / 'results/v3/freeze_manifest.json',
            'schema': location / 'configs/fields.yaml',
            'relation_config': location / 'configs/v3_relations.yaml',
            'runner_source': location / 'src/paper1_active_req/experiment/reviewer_runner.py',
            'metrics_source': location / 'src/paper1_active_req/evaluation/reviewer_metrics.py',
        }
        freeze = json.loads(inputs['formal_freeze'].read_text(encoding='utf-8'))
        for source in freeze['source_hashes']:
            normalized = source.replace('\\', '/')
            if '/src/' not in normalized:
                raise ValueError('unexpected legacy source layout')
            relative = 'src/' + normalized.split('/src/', 1)[1]
            inputs['legacy_source:' + relative] = location / relative
        return inputs

    reviewer_runner.protected_inputs = relocated_inputs
    summarize_reviewer_supplement.protected_inputs = relocated_inputs
    import verify_reviewer_records

    def check_existing(path, payload):
        # 发布包自带历史验证结果；重复核验只能逐字段相等，不能覆盖。
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError('发布包缺少原始验证记录: ' + str(path))
        if json.loads(path.read_text(encoding='utf-8')) != payload:
            raise ValueError('本次复算与原始验证记录不一致')
        print(json.dumps({'status': 'passed', 'record_count': payload['record_count'],
                          'step_count': payload['step_count'],
                          'metric_value_count': payload['metric_value_count'],
                          'existing_verification_matches': True}))

    verify_reviewer_records.write_once = check_existing
    verify_reviewer_records.main()


if __name__ == '__main__':
    main()
