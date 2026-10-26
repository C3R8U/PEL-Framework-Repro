import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]

def load_script(name):
    file=ROOT/'scripts'/name
    spec=importlib.util.spec_from_file_location(name.replace('.py',''),file)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def paired_fixture():
    rows=[]
    for teacher in range(4):
        for group in ['experimental','control']:
            for i in range(7):
                sid=f't{teacher}-{group}-{i}'
                cls=f'{teacher}-{group}'
                base=50 + teacher + i*0.25
                change=12+teacher if group=='experimental' else 4 +teacher*.5
                for time,val in [('pre',base),('post',base+change+i*.07)]:
                    rows.append(dict(student_id=sid,class_id=cls,teacher_id=teacher,
                        group=group,time=time,vocabulary=val))
    return pd.DataFrame(rows)


def test_four_teacher_paired_inference():
    m=load_script('analyze_classroom.py')
    res=m.paired_class_analysis(paired_fixture(),'vocabulary')
    assert res['class_count']==8
    assert res['df']==3
    assert res['exact_two_sided_pair_flip_p']==pytest.approx(.125)
    assert res['class_pair_adjusted_change_difference']>0


def test_crossover_six_conditions_repeated():
    m=load_script('analyze_ablation.py')
    conditions=['full','no_profile','no_generation','no_path','generic_ai','equal_practice_digital']
    rng=np.random.default_rng(4)
    rows=[]
    for i in range(12):
        for k,c in enumerate(conditions):
            rows.append({'student_id':f'S{i:02}','condition':c,'period':k+1,
              'task_version':f'T{k+1}', 'vocabulary':80-2*k+i*.11+rng.normal(0,2)})
    result=m.analyze(pd.DataFrame(rows),'vocabulary',conditions)
    assert result['n_subjects']==12
    assert len(result['comparisons'])==5
    assert 0 <= result['repeated_anova']['partial_eta_squared'] <= 1


def test_teacher_rating_quality():
    m=load_script('evaluate_exercise_quality.py')
    dims=['grammatical_correctness','pedagogical_relevance','cefr_alignment','answer_validity']
    rows=[]
    for task in range(12):
        for strat in ('profile_guided','generic_ai','human'):
            for r in range(5):
                vals={d:int((task*3+r+(strat=='human'))%9!=0) for d in dims}
                rows.append({'task_id':task,'strategy':strat,'rater_id':r,**vals})
    result=m.evaluate(pd.DataFrame(rows),dims,['profile_guided','generic_ai','human'])
    assert result['n_tasks_per_strategy']==12
    assert 'fleiss_kappa' in result['dimensions']['answer_validity']


def test_itt_multiple_imputation_preserves_cohort_size():
    mod=load_script('analyze_itt.py')
    d=paired_fixture()
    d['grammar_error_rate']=34 - d['vocabulary']*.05
    d['fluency']=d['vocabulary']*.9
    # A missing post outcome remains an assigned learner, not dropped.
    d.loc[(d.student_id.str.endswith('-0')) & (d.time=='post'),['vocabulary','grammar_error_rate','fluency']]=np.nan
    out=mod.multiple_imputation(d,draws=2)
    assert out['n_randomized']==56
    assert out['imputations']==2
    assert len(out['results']['vocabulary']['conservative_t3_interval'])==2
