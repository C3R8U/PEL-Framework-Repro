"""Blind 5-rater assessment: item-level majority, Fleiss kappa, paired strategy contrasts."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa
from scipy.stats import ttest_rel
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pel_repro.common import require_columns,read_json,save_json

def evaluate(df,dimensions,strategies,expected_n=None):
    require_columns(df,['task_id','strategy','rater_id']+dimensions)
    if df.duplicated(['task_id','strategy','rater_id']).any():raise ValueError('Duplicate rater for task-strategy')
    if set(df.strategy)!=set(strategies):raise ValueError('All three strategies required')
    combos=df.groupby(['task_id','strategy']).rater_id.nunique()
    if (combos!=5).any():raise ValueError('Exactly 5 raters per question per strategy')
    n_task=df.task_id.nunique()
    if expected_n and n_task!=expected_n:raise ValueError('Expected 200 paired tasks')
    if len(combos)!=n_task*len(strategies):raise ValueError('Same tasks required across three strategies')
    if df.groupby(['task_id','strategy']).rater_id.apply(lambda x:tuple(sorted(x.astype(str)))).nunique()!=1:
        raise ValueError('Use the same five independent raters throughout')
    output={'n_tasks_per_strategy':n_task,'n_rating_rows':len(df),'dimensions':{}}
    for dim in dimensions:
        if not set(df[dim].unique()).issubset({0,1}):raise ValueError('Binary valid ratings required')
        majority=df.groupby(['task_id','strategy'])[dim].mean().ge(.6).reset_index(name='passed')
        by=majority.groupby('strategy').passed.agg(['mean','sum','size'])
        table=[]
        for (_,grp) in df.groupby(['task_id','strategy']):
            ok=int(grp[dim].sum());table.append([5-ok,ok])
        k=float(fleiss_kappa(np.asarray(table),method='fleiss'))
        wide=majority.pivot(index='task_id',columns='strategy',values='passed')
        test=ttest_rel(wide['profile_guided'].astype(float),wide['generic_ai'].astype(float))
        differences=(wide['profile_guided'].astype(float)-wide['generic_ai'].astype(float)).to_numpy()
        rng=np.random.default_rng(42)
        boot=np.asarray([rng.choice(differences,size=len(differences),replace=True).mean() for _ in range(2000)])
        bounds=(100*np.quantile(boot,[.025,.975])).tolist()
        output['dimensions'][dim]={'strategy_rates':by.to_dict('index'),'fleiss_kappa':k,
           'guided_minus_generic_percentage_points':100*float(differences.mean()),'paired_bootstrap_ci95_pp':bounds,
           'exploratory_paired_test_p':float(test.pvalue)}
    return output

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',default='results/quality.json');a=p.parse_args()
 cfg=read_json(ROOT/'configs/components.json')['content_quality'];df=pd.read_csv(a.input)
 res=evaluate(df,cfg['dimensions'],cfg['strategies'],expected_n=cfg['tasks'])
 save_json(res,ROOT/a.output);print('Saved',ROOT/a.output)
if __name__=='__main__':main()
