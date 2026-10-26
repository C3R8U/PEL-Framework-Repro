"""Split authorized inputs by learner without cross-partition identity overlap."""
import argparse
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pel_repro.common import group_split,require_columns,save_json,sha256_file,ensure_private_source

p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output-dir',required=True)
p.add_argument('--group-column',default='learner_id');p.add_argument('--ratio',nargs=3,type=float,default=[.7,.2,.1]);p.add_argument('--seed',type=int,default=42)
a=p.parse_args();source=ensure_private_source(a.input);d=pd.read_csv(source)
sets=group_split(d,a.group_column,a.seed,a.ratio)
out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
summary={'source_sha256':sha256_file(source),'seed':a.seed,'ratio':a.ratio,'group_column':a.group_column,'slices':{}}
for k,df in sets.items():
    df.to_csv(out/(k+'.csv'),index=False)
    summary['slices'][k]={'rows':len(df),'learners':df[a.group_column].nunique()}
save_json(summary,out/'split_manifest.json');print(summary)
