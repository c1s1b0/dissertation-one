"""核对发布清单，不更改任何实验文件。"""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'SHA256SUMS.json').read_text(encoding='utf-8'))
bad=[name for name,digest in manifest.items() if not (root/name).is_file() or hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest]
if bad: raise SystemExit('校验失败: '+str(bad))
print('发布文件校验通过:',len(manifest))
