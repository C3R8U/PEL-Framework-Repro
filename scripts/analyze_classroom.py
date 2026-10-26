"""Eight classroom clusters: fixed effects, paired-class inference, exact randomization.

This does not fabricate data or treat 235 learners as 235 independent clusters.
The mixed-effects fixed-effect p-values are asymptotic; the 4-pair exact randomization
p-value is reported independently and is constrained by only 16 assignments.
"""
import argparse
import itertools
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import t as tdist
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pel_repro.common import require_columns,save_json


def paired_class_analysis(df, measure):
    require_columns(df,['student_id','class_id','teacher_id','group','time',measure])
    q=df.copy()
    if not set(q.group).issubset({'experimental','control'}):raise ValueError('Invalid groups')
    if not set(q.time).issubset({'pre','post'}):raise ValueError('Invalid time')
    # Complete-case student-level change; 1 pre and 1 post per retained student.
    if q.duplicated(['student_id','time']).any():raise ValueError('Duplicate student-time records')
    idx=['student_id','class_id','teacher_id','group']
    pivot=q.pivot(index=idx,columns='time',values=measure).dropna()
    if not {'pre','post'}.issubset(pivot):raise ValueError('Need pre and post records')
    delta=(pivot['post']-pivot['pre']).rename('change').reset_index()
    classes=delta.groupby(['teacher_id','group','class_id'],as_index=False).change.mean()
    teacher_sets=classes.groupby('teacher_id').group.apply(set)
    if len(teacher_sets)!=4 or any(v!={'experimental','control'} for v in teacher_sets):
        raise ValueError('Expected four teachers, each with one class per group')
    if classes.class_id.nunique()!=8: raise ValueError('Expected eight classes')
    mean=classes.pivot(index='teacher_id',columns='group',values='change')
    pairdiff=(mean['experimental']-mean['control']).to_numpy()
    estimate=float(pairdiff.mean()); n=len(pairdiff)
    se=float(pairdiff.std(ddof=1)/np.sqrt(n))
    if se==0: raise ValueError('Zero paired-class variance; report exact nonparametric separately')
    critical=tdist.ppf(0.975,n-1)
    tstat=estimate/se
    flips=np.asarray([np.sum(pairdiff*np.array(signs))/n for signs in itertools.product([-1,1],repeat=n)])
    exact=float(np.mean(np.abs(flips)>=abs(estimate)-1e-12))
    return {'complete_students':int(len(delta)),'class_count':int(len(classes)),
      'n_teacher_pairs':n,'class_pair_adjusted_change_difference':estimate,
      'paired_cluster_t':float(tstat),'df':n-1,'p_pair_t':float(2*tdist.sf(abs(tstat),df=n-1)),
      'ci95_pair_t':[float(estimate-critical*se),float(estimate+critical*se)],
      'exact_two_sided_pair_flip_p':exact,
      'caution':'Paired-class inference uses FOUR independent pairs; an exact two-sided test cannot reach 0.05 with only 4 pairs.'}


def fit_mixedlm(df,measure):
    """Exploratory variance components / G×T, not valid small-cluster standalone inference."""
    import statsmodels.formula.api as smf
    d=df.dropna(subset=[measure]).copy()
    d['tpost']=(d['time']=='post').astype(int)
    d['gexp']=(d['group']=='experimental').astype(int)
    d['class_id']=d.class_id.astype(str);d['student_id']=d.student_id.astype(str)
    model=smf.mixedlm(f'{measure} ~ gexp*tpost',data=d,groups=d.class_id,
                      re_formula='1',vc_formula={'student':'0+C(student_id)'})
    fit=model.fit(reml=True,method='lbfgs',maxiter=1000,disp=False)
    b=float(fit.params['gexp:tpost']);se=float(fit.bse['gexp:tpost'])
    var_class=float(fit.cov_re.iloc[0,0]);var_student=float(fit.vcomp[0]) if len(fit.vcomp) else 0
    variance=float(fit.scale);icc=var_class/(var_class+var_student+variance)
    return {'interaction':b,'standard_error':se,'wald_ci95_approx':[b-1.96*se,b+1.96*se],
      'class_icc':icc,'n_obs':int(fit.nobs),'warning':'Wald interval is approximate and NOT Satterthwaite / Kenward-Roger corrected. Prefer paired-cluster and exact inference for 8 classes.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True)
    p.add_argument('--out',default='results/classroom_stats.json');p.add_argument('--mixed-model',action='store_true')
    a=p.parse_args();df=pd.read_csv(a.input)
    results={}
    for measure in ['vocabulary','grammar_error_rate','fluency']:
        results[measure]={'paired_class':paired_class_analysis(df,measure)}
        if a.mixed_model:
            try:results[measure]['mixed_effects']=fit_mixedlm(df,measure)
            except Exception as exc:results[measure]['mixed_model_error']=str(exc)
    save_json(results,ROOT/a.out);print('Saved statistics to',ROOT/a.out)
if __name__=='__main__':main()
