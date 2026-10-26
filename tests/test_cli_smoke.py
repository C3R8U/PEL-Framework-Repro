from pathlib import Path
import subprocess
import sys
import json

ROOT=Path(__file__).resolve().parents[1]


def test_run_model_metadata_without_network():
    cp=subprocess.run([sys.executable,str(ROOT/'scripts/verify_model.py'),
             '--report','results/test_model.json'],cwd=ROOT,capture_output=True,text=True,check=True)
    assert 'revision_format_only' in cp.stdout


def test_path_planner_without_network():
    cp=subprocess.run([sys.executable,str(ROOT/'scripts/optimize_path.py'),
             '--output','results/test_path.json'],cwd=ROOT,capture_output=True,text=True,check=True)
    assert 'common_words' in cp.stdout
