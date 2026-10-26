import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require_columns(df, names):
    absent = sorted(set(names) - set(df.columns))
    if absent:
        raise ValueError(f'Missing required CSV columns: {absent}')


def ensure_private_source(path):
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f'Input data must be supplied separately: {p}')
    return p


def group_split(df, group_col='learner_id', seed=42, ratios=(0.7, 0.2, 0.1)):
    from sklearn.model_selection import train_test_split
    require_columns(df, [group_col])
    if any(pd.isna(df[group_col])):
        raise ValueError('Missing learner identity prevents leakage-safe splitting')
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError('Ratios must sum to 1')
    groups = np.asarray(sorted(df[group_col].astype(str).unique()))
    if len(groups) < 10:
        raise ValueError('Need >=10 independent learners to split safely')
    train_ids, remain_ids = train_test_split(groups, test_size=1-ratios[0], random_state=seed)
    val_ids, test_ids = train_test_split(remain_ids, test_size=ratios[2]/sum(ratios[1:]), random_state=seed)
    a = df[group_col].astype(str)
    sets = {
        'train': df[a.isin(train_ids)].copy(),
        'validation': df[a.isin(val_ids)].copy(),
        'test': df[a.isin(test_ids)].copy(),
    }
    if len(set(train_ids)&set(val_ids)) or len(set(train_ids)&set(test_ids)) or len(set(val_ids)&set(test_ids)):
        raise AssertionError('Identity leakage')
    return sets
