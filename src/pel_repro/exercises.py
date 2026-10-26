"""Deterministic prompt serialization, Qwen inference, structural checks, provenance."""
import json
from pathlib import Path

from .common import read_json, save_json, set_seed, sha256_file


OPTIONS = set('ABCD')


def validate_exercises(payload, expected_count):
    if set(payload) != {'exercises'} or not isinstance(payload['exercises'], list):
        raise ValueError('JSON must have a single `exercises` list field')
    if len(payload['exercises']) != expected_count:
        raise ValueError('Exercise count differs from requested count')
    for index, row in enumerate(payload['exercises'], 1):
        if set(row) != {'question', 'options', 'answer', 'hint'}:
            raise ValueError(f'Exercise {index} has missing/extra fields')
        if not isinstance(row['question'], str) or not row['question'].strip():
            raise ValueError(f'Exercise {index} question is empty')
        if not isinstance(row['hint'], str):
            raise ValueError(f'Exercise {index} missing text hint')
        if not isinstance(row['options'], dict) or set(row['options']) != OPTIONS:
            raise ValueError(f'Exercise {index} must have four lettered options')
        if row['answer'] not in OPTIONS:
            raise ValueError(f'Exercise {index} answer must be A/B/C/D')
        choices = list(row['options'].values())
        if any(not isinstance(c, str) or not c.strip() for c in choices):
            raise ValueError(f'Exercise {index} contains blank choices')
        if len(set(s.strip().casefold() for s in choices)) != 4:
            raise ValueError(f'Exercise {index} contains repeated options')
    return payload


def render_messages(root, count, words, level, learner_profile):
    if not isinstance(count, int) or count < 1 or count > 20:
        raise ValueError('count must be 1..20')
    if level not in {'A1','A2','B1','B2','C1','C2'}:
        raise ValueError('Invalid CEFR level')
    system_path = Path(root) / 'prompts' / 'exercise_system.txt'
    user_path = Path(root) / 'prompts' / 'exercise_user.txt'
    system = system_path.read_text(encoding='utf-8').strip()
    user = user_path.read_text(encoding='utf-8').format(
        count=count, target_words=', '.join(words), cefr_level=level,
        learner_profile=json.dumps(learner_profile, ensure_ascii=False, sort_keys=True))
    return [{'role': 'system', 'content':system}, {'role':'user','content':user}], {
        'system_sha256': sha256_file(system_path), 'user_sha256':sha256_file(user_path)}


def run_qwen(root, count, words, level, profile, output_path):
    # Not executed by tests: downloads ~15 GB from Hugging Face on first run.
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    cfg = read_json(Path(root) / 'configs' / 'model.json')
    set_seed(cfg['generation']['seed'])
    messages, hashes = render_messages(root, count, words, level, profile)
    model_id, revision = cfg['model_id'], cfg['revision']
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    dtype_name = cfg['generation']['torch_dtype']
    if dtype_name == 'bfloat16' and not torch.cuda.is_available():
        raise RuntimeError('bfloat16 A100/CUDA expected; configure dtype/hardware before real inference')
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
        trust_remote_code=False, torch_dtype=getattr(torch, dtype_name), device_map='auto')
    model.eval()
    inp = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                  return_tensors='pt').to(model.device)
    gen = cfg['generation']
    with torch.no_grad():
        out = model.generate(inp, do_sample=True, temperature=gen['temperature'],
            top_p=gen['top_p'], max_new_tokens=gen['max_new_tokens'],
            repetition_penalty=gen['repetition_penalty'], pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0][inp.shape[-1]:], skip_special_tokens=True).strip()
    if raw.startswith('```'):
        raw = raw.strip('`').lstrip('json').strip()
    parsed = validate_exercises(json.loads(raw), count)
    save_json({'exercises':parsed['exercises'], 'provenance':{
        'model_id':model_id, 'revision':revision, 'prompt_sha256':hashes,
        'generation':gen, 'automated_structural_check_only':True,
        'human_quality_validation_required':True}}, output_path)
    return output_path
