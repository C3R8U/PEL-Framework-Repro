"""Behavior pseudo-clustering and GaussianNB. No claim of true style labels."""
import argparse
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, accuracy_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pel_repro.common import read_json,require_columns,save_json

p=argparse.ArgumentParser()
p.add_argument('--split-dir',required=True);p.add_argument('--output-dir',default='results/behavior');a=p.parse_args()
frames={s:pd.read_csv(Path(a.split_dir)/(s+'.csv')) for s in ['train','validation','test']}
features=read_json(ROOT/'configs/components.json')['behavior_preferences']['features']
for df in frames.values(): require_columns(df, ['learner_id']+features)
scaler=StandardScaler().fit(frames['train'][features])
tr=scaler.transform(frames['train'][features]); kmeans=KMeans(n_clusters=4,n_init=20,random_state=42).fit(tr)
y_train=kmeans.predict(tr); clf=GaussianNB(var_smoothing=1e-9).fit(tr,y_train)
report={'interpretation':'agreement with unsupervised pseudo-labels; NOT ground-truth learning style accuracy'}
for split,df in frames.items():
    x=scaler.transform(df[features]); pseudo=kmeans.predict(x)
    predicted=clf.predict(x)
    report[split]={'n':len(df),'pseudo_label_agreement':float(accuracy_score(pseudo,predicted))}
output=Path(a.output_dir);output.mkdir(parents=True,exist_ok=True)
joblib.dump({'scaler':scaler,'kmeans':kmeans,'classifier':clf,'features':features},output/'models.joblib')
save_json(report,output/'metrics.json');print(report)
