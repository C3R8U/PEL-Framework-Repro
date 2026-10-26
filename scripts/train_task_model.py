"""Predict task correctness from licensed or consented response records."""
import argparse
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pel_repro.common import require_columns,save_json
p=argparse.ArgumentParser();p.add_argument('--split-dir',required=True);p.add_argument('--output-dir',default='results/task_model');a=p.parse_args()
features=['response_time_seconds','item_difficulty']
fr={s:pd.read_csv(Path(a.split_dir)/(s+'.csv')) for s in ('train','validation','test')}
for x in fr.values():
 require_columns(x,['learner_id','task_type','correct']+features)
 if not set(x.correct.dropna().unique()).issubset({0,1}):raise ValueError('correct must be binary')
def xform(frame):
 m=frame[features].copy()
 m['is_listening']=(frame['task_type']=='listening').astype(int)
 if frame[features].isna().any().any():raise ValueError('Complete case response features required')
 return m
model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42))
model.fit(xform(fr['train']),fr['train']['correct'])
out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
report={'warning':'This requires ACTUAL student task responses; public TOEFL question PDFs do not contain these records.'}
for split,frame in fr.items():
 pred=model.predict(xform(frame))
 report[split]={'n':len(frame),'accuracy':float(accuracy_score(frame.correct,pred)),'weighted_f1':float(f1_score(frame.correct,pred,average='weighted',zero_division=0))}
joblib.dump(model,out/'model.joblib');save_json(report,out/'metrics.json');print(report)
