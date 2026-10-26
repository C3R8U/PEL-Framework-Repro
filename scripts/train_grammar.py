"""Train supervised grammar error risk model on authentically labeled items."""
import argparse
from pathlib import Path
import sys
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,precision_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pel_repro.common import require_columns,save_json
p=argparse.ArgumentParser();p.add_argument('--split-dir',required=True);p.add_argument('--output-dir',default='results/grammar');a=p.parse_args()
col=['historical_error_rate','proficiency_score','item_complexity','transfer_risk']
fr={s:pd.read_csv(Path(a.split_dir)/(s+'.csv')) for s in ('train','validation','test')}
for d in fr.values():require_columns(d,['learner_id','error_label']+col)
model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42))
model.fit(fr['train'][col],fr['train'].error_label)
t=fr['test'];prob=model.predict_proba(t[col])[:,1];pred=model.predict(t[col]);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
rep={'test_n':len(t),'auc':float(roc_auc_score(t.error_label,prob)),'precision':float(precision_score(t.error_label,pred,zero_division=0)),'source_required':'error_label MUST be independently annotated'}
joblib.dump(model,out/'model.joblib');save_json(rep,out/'metrics.json');print(rep)
