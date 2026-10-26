"""ITT sensitivity: multiple imputation of missing POST outcomes, then 4-pair inference.

Exploratory MAR analysis. Requires ALL assigned students, including dropouts,
with pre and optionally missing post rows. Fully joint multilevel MI is not
implemented; fixed class indicators and 4-pair estimates reduce, but do not
eliminate, limitations. Keep result clearly labeled as sensitivity.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pel_repro.common import require_columns,save_json

MEASURES=['vocabulary','grammar_error_rate','fluency']


def multiple_imputation(df, draws=20, seed=42):
    require_columns(df,['student_id','class_id','teacher_id','group','time']+MEASURES)
    if set(df.group)!={'experimental','control'}: raise ValueError('Require both arms')
    if df.duplicated(['student_id','time']).any(): raise ValueError('Duplicate student-time')
    if df.student_id.nunique() < 16: raise ValueError('Need more learner records')
    pivot=df.pivot(index=['student_id','class_id','teacher_id','group'],columns='time',values=MEASURES)
    if any((k,'pre') not in pivot or (k,'post') not in pivot for k in MEASURES):
        raise ValueError('Both pre/post rows are required, though post results may be missing')
    if any(pivot[(k,'pre')].isna().any() for k in MEASURES):
        raise ValueError('Only post-outcome missingness is supported. Pre must be observed.')
    info=pivot.reset_index()[['student_id','class_id','teacher_id','group']]
    numeric=pivot.copy()
    numeric.columns=[f'{metric}_{time}' for metric,time in numeric.columns]
    design=pd.get_dummies(info[['class_id']],prefix='cluster',dtype=float)
    design['gexp']=(info.group=='experimental').astype(float)
    features=pd.concat([numeric.reset_index(drop=True), design],axis=1)
    out=[]
    for i in range(draws):
        model=IterativeImputer(max_iter=15,random_state=seed+i,sample_posterior=True,
                               skip_complete=False)
        imp=pd.DataFrame(model.fit_transform(features),columns=features.columns)
        long=info.copy()
        for k in MEASURES:
            long[f'{k}_change']=imp[f'{k}_post'].to_numpy()-imp[f'{k}_pre'].to_numpy()
        means={}
        for k in MEASURES:
            c=long.groupby(['teacher_id','group','class_id'],as_index=False)[f'{k}_change'].mean()
            teacher=c.groupby('teacher_id').group.apply(set)
            if len(teacher)!=4 or any(x!={'experimental','control'} for x in teacher) or c.class_id.nunique()!=8:
                raise ValueError('Must have exactly 4 teacher-paired experimental/control class pairs')
            wide=c.pivot(index='teacher_id',columns='group',values=f'{k}_change')
            diff=(wide['experimental']-wide['control']).to_numpy()
            means[k]=(float(diff.mean()), float(diff.var(ddof=1)/4))
        out.append(means)
    report={'imputations':draws,'n_randomized':int(df.student_id.nunique()),
            'assumption':'MAR conditional on recorded group, class and baseline outcomes; Bayesian iterative imputation, not hierarchical joint MI',
            'results':{}}
    for k in MEASURES:
        b=np.array([d[k][0] for d in out]);u=np.array([d[k][1] for d in out])
        est=float(b.mean());total=float(u.mean()+(1+1/draws)*b.var(ddof=1))
        se=float(np.sqrt(total));crit=float(tdist.ppf(.975,3))
        report['results'][k]={'pooled_effect':est,'SE_Rubin':se,
          'conservative_t3_interval':[est-crit*se,est+crit*se],
          'limitations':'Not an exact randomization CI; four-pair degree-of-freedom bound is conservative only.'}
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True)
    p.add_argument('--draws',type=int,default=20)
    p.add_argument('--output',default='results/itt_sensitivity.json');a=p.parse_args()
    if a.draws < 2: raise ValueError('Need >=2 independent imputations')
    output=multiple_imputation(pd.read_csv(a.input),a.draws)
    save_json(output,ROOT/a.output);print('Saved',ROOT/a.output)
if __name__=='__main__': main()
