"""Within-subject six-condition evaluation with repeated ANOVA + Bonferroni."""
import argparse
import itertools
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, f as f_dist
from statsmodels.stats.anova import AnovaRM
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pel_repro.common import require_columns,save_json,read_json

def analyze(df,metric,conditions):
    require_columns(df,['student_id','condition','period','task_version',metric])
    sub=df[['student_id','condition','period','task_version',metric]].dropna()
    if sub.duplicated(['student_id','condition']).any():raise ValueError('Duplicate subject-condition')
    full=sub.groupby('student_id').condition.apply(lambda x:set(x)==set(conditions))
    if not full.all():raise ValueError('Need all six conditions for each included student, otherwise preregister incomplete-case handling')
    if not sub.groupby('student_id').period.nunique().eq(len(conditions)).all():raise ValueError('Each condition must occur in a distinct period per learner')
    if not sub.groupby('student_id').task_version.nunique().eq(len(conditions)).all():raise ValueError('Use different task versions within learner')
    summary=sub.groupby('condition')[metric].agg(['mean','std','count']).to_dict('index')
    fit=AnovaRM(sub,depvar=metric,subject='student_id',within=['condition']).fit()
    r=fit.anova_table.loc['condition'];F=float(r['F Value']);df1=float(r['Num DF']);df2=float(r['Den DF'])
    eta=(F*df1)/(F*df1+df2)
    # Greenhouse-Geisser sphericity correction for six repeated conditions.
    cov = sub.pivot(index='student_id',columns='condition',values=metric).reindex(columns=conditions).cov().to_numpy()
    k=len(conditions)
    centering=np.eye(k)-np.ones((k,k))/k
    cs=centering@cov@centering
    epsilon=float((np.trace(cs)**2)/((k-1)*np.trace(cs@cs)))
    epsilon=float(np.clip(epsilon,1/(k-1),1))
    corrected_df1,corrected_df2=epsilon*df1,epsilon*df2
    wide=sub.pivot(index='student_id',columns='condition',values=metric)
    tests=[]
    for comp in conditions:
        if comp=='full':continue
        t,p=ttest_rel(wide['full'],wide[comp])
        tests.append({'comparison':f'full_vs_{comp}','mean_difference':float((wide['full']-wide[comp]).mean()),
            'paired_t':float(t),'df':len(wide)-1,'p_raw':float(p),'bonferroni_p':min(float(p)*5,1.0)})
    return {'n_subjects':len(wide),'summary':summary,'repeated_anova':{'F':F,'df1':df1,'df2':df2,'p':float(r['Pr > F']),'partial_eta_squared':eta,'gg_epsilon':epsilon,'gg_df1':corrected_df1,'gg_df2':corrected_df2,'p_gg':float(f_dist.sf(F, corrected_df1, corrected_df2))},'comparisons':tests,
    'limitations':'GG sphericity correction is included; cross-over period/task-version confounding must also be assessed with an appropriate mixed model. This does not establish 16-week module-specific causal effects.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True)
    p.add_argument('--output',default='results/ablation.json');a=p.parse_args()
    df=pd.read_csv(a.input);c=read_json(ROOT/'configs/components.json')['ablation']['conditions']
    result={k:analyze(df,k,c) for k in ['vocabulary','grammar_accuracy','fluency','task_satisfaction']}
    save_json(result,ROOT/a.output);print('Saved',ROOT/a.output)
if __name__=='__main__':main()
