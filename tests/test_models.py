import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pel_repro.common import group_split
from pel_repro.learning import weighted_affect, update_profile, optimize_path
from pel_repro.exercises import render_messages,validate_exercises

ROOT=Path(__file__).resolve().parents[1]


def test_immutable_model_revision_and_architecture_metadata():
    cfg=json.loads((ROOT/'configs/model.json').read_text())
    assert cfg['revision']=='4709f6c0771f0185a675b046268cdc1d1f2c74ce'
    assert cfg['architecture_expected']['num_hidden_layers']==28
    assert cfg['architecture_expected']['num_key_value_heads']==4
    assert cfg['architecture_expected']['vocab_size']==152064


def test_group_splitting_blocks_learner_leakage():
    df=pd.DataFrame({'learner_id':np.repeat(np.arange(30),2),'text':['example']*60})
    s=group_split(df,seed=42)
    ids={name:set(part.learner_id) for name,part in s.items()}
    assert ids['train'].isdisjoint(ids['validation'])
    assert ids['train'].isdisjoint(ids['test'])
    assert ids['validation'].isdisjoint(ids['test'])
    assert sum(len(x) for x in s.values())==60


def test_qwen_prompt_and_json_constraints():
    msgs,h=render_messages(ROOT,1,['adapt'],'B1',{'skill':'vocabulary'})
    assert len(h['system_sha256'])==64
    assert 'B1' in msgs[-1]['content']
    exercise={'exercises':[{'question':'Fill the blank','options':{'A':'a','B':'b','C':'c','D':'d'},'answer':'B','hint':'context'}]}
    assert validate_exercises(exercise,1)==exercise
    exercise['exercises'][0]['options']['D']='a'
    with pytest.raises(ValueError):validate_exercises(exercise,1)


def test_affect_missing_modality_and_profile_update():
    assert weighted_affect(.8,None)==pytest.approx(.8)
    assert weighted_affect(.8,.2)==pytest.approx(.53)
    assert update_profile([.2,.4],[.8,.6],.5)==pytest.approx([.5,.5])
    with pytest.raises(ValueError):weighted_affect(None,None)


def test_prerequisite_path():
    tasks=json.loads((ROOT/'examples/learning_tasks.json').read_text())
    weights=json.loads((ROOT/'configs/components.json').read_text())['path']
    result=optimize_path(tasks,weights)
    assert result['ordered_task_ids'][0]=='common_words'
    assert set(result['ordered_task_ids'])=={'common_words','past_tense','dialogue'}
    assert result['total_cost']>=0
