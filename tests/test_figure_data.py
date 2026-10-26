import pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_summary_means_and_no_invented_uncertainty():
    d=pd.read_csv(ROOT/'data/public_summary/figures_4_8.csv')
    assert 'sd' not in d.columns
    def get(fig,indicator,group,time):
        x=d[(d.figure==fig)&(d.indicator==indicator)&(d.group==group)&(d.time==time)]
        return float(x.value.iloc[0])
    assert get(4,'vocabulary','experimental','post')==84.0
    assert get(5,'grammar_error_rate','control','pre')==31.8
    assert get(6,'pause_frequency','experimental','pre')==12.3
    assert get(7,'satisfaction_A','control','post')==78.5
    assert get(8,'delayed_retention','experimental','post')==82.7
