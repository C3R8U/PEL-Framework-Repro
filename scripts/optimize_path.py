import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pel_repro.common import read_json,save_json
from pel_repro.learning import optimize_path
p=argparse.ArgumentParser();p.add_argument('--tasks',default='examples/learning_tasks.json');p.add_argument('--output',default='results/path.json');a=p.parse_args()
tasks=read_json(ROOT/a.tasks)
w=read_json(ROOT/'configs/components.json')['path']
ans=optimize_path(tasks,w)
save_json(ans,ROOT/a.output);print(ans)
