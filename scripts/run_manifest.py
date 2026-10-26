"""Write machine-readable execution provenance without copying any input data."""
import argparse
from datetime import datetime,timezone
import importlib.metadata as metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pel_repro.common import sha256_file,save_json,read_json

p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='*',default=[])
p.add_argument('--output',default='results/run_manifest.json');a=p.parse_args()
versions={}
for name in ['numpy','pandas','scipy','scikit-learn','statsmodels','torch','transformers','accelerate','huggingface-hub']:
 try:versions[name]=metadata.version(name)
 except metadata.PackageNotFoundError: versions[name]='not_installed'
try:git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,stderr=subprocess.DEVNULL,text=True).strip()
except Exception:git=None
inputs={}
for s in a.inputs:
 f=Path(s)
 if not f.is_file():raise FileNotFoundError(f)
 # file name only: avoid revealing private path or raw content
 inputs[f.name]=sha256_file(f)
report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),
 'python':sys.version,'platform':platform.platform(),'dependencies':versions,'git_sha':git,
 'model_revision':read_json(ROOT/'configs/model.json')['revision'],
 'input_sha256_no_raw_data':inputs,'warning':'This manifest is a provenance record, not a claim that manuscript results are independently reproduced.'}
save_json(report,ROOT/a.output);print(json.dumps(report,ensure_ascii=False,indent=2))
