import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pel_repro.exercises import run_qwen

p=argparse.ArgumentParser()
p.add_argument('--words',nargs='+',required=True)
p.add_argument('--cefr',choices=['A1','A2','B1','B2','C1','C2'],required=True)
p.add_argument('--count',type=int,default=10)
p.add_argument('--profile-json',default='{}')
p.add_argument('--output',default='results/exercises.json')
a=p.parse_args()
import json
profile=json.loads(a.profile_json)
print('Output saved:',run_qwen(ROOT,a.count,a.words,a.cefr,profile,ROOT/a.output))
