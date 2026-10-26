"""Check an immutable official Qwen revision and optionally fetch only config.json."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pel_repro.common import read_json, save_json


def verify_file(data, expected):
    bad={key: {'expected':value,'actual':data.get(key)} for key,value in expected.items()
         if data.get(key)!=value}
    if bad: raise ValueError(f'Model architecture mismatch: {bad}')
    return True


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--online',action='store_true',help='Download only official pinned config.json')
    p.add_argument('--config-json',help='Offline config.json path for verification')
    p.add_argument('--report',default='results/model_verification.json')
    a=p.parse_args()
    meta=read_json(ROOT/'configs/model.json')
    revision=meta['revision']
    if len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Revision must be a 40-character SHA')
    report={'repo':meta['model_id'],'revision':revision,'architecture_expected':meta['architecture_expected'],'verification':'not_performed'}
    if a.online:
        from huggingface_hub import hf_hub_download
        file=hf_hub_download(repo_id=meta['model_id'], filename='config.json', revision=revision)
        data=read_json(file)
        verify_file(data,meta['architecture_expected'])
        report['verification']='official_hub_config_matches'
    elif a.config_json:
        data=read_json(a.config_json)
        verify_file(data,meta['architecture_expected'])
        report['verification']='user_supplied_config_matches'
    else:
        report['verification']='revision_format_only; use --online to verify official config'
    save_json(report,ROOT/a.report)
    print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
