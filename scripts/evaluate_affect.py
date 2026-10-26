"""Evaluate aligned text/audio scores against independent HUMAN affect labels."""
import argparse
from pathlib import Path
import sys
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pel_repro.common import read_json,save_json,require_columns,ensure_private_source
from pel_repro.learning import weighted_affect

p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',default='results/affect_evaluation.json');a=p.parse_args()
df=pd.read_csv(ensure_private_source(a.input));require_columns(df,['learner_id','task_id','text_score','audio_score','human_label'])
w=read_json(ROOT/'configs/components.json')['affect']['weights']
labelmap={0:'negative',1:'neutral',2:'positive'}
truth=df['human_label'].astype(str).tolist()
if any(t not in labelmap.values() for t in truth):raise ValueError('Human labels must be negative/neutral/positive')
scores=[weighted_affect(t if pd.notna(t) else None,a if pd.notna(a) else None,w['text'],w['audio']) for t,a in zip(df.text_score,df.audio_score)]
pred=[labelmap[0 if score<1/3 else 1 if score<2/3 else 2] for score in scores]
res={'n':len(df),'accuracy':accuracy_score(truth,pred),'macro_f1':f1_score(truth,pred,labels=list(labelmap.values()),average='macro',zero_division=0),'confusion_matrix':confusion_matrix(truth,pred,labels=list(labelmap.values())).tolist(),'labels':list(labelmap.values()),'warning':'Score thresholds are explicitly defined assumptions. Models producing text_score/audio_score must be documented separately.'}
save_json(res,ROOT/a.output);print(res)
