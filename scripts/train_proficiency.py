"""Fine-tune pinned BERT on 16 ORIGINAL proficiency-level labels.

Requires authorized split CSV files, a GPU, and `requirements-model.txt`.
Outputs are local, never bundled in the publication ZIP.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pel_repro.common import read_json,require_columns,save_json,set_seed


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--split-dir',required=True)
    ap.add_argument('--output-dir',default='results/bert_proficiency')
    ap.add_argument('--epochs',type=int)
    a=ap.parse_args()
    from sklearn.metrics import accuracy_score,f1_score
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,Trainer,TrainingArguments
    import torch
    cfg=read_json(ROOT/'configs/components.json')['proficiency']
    set_seed(42)
    data={s:pd.read_csv(Path(a.split_dir)/(s+'.csv')) for s in ('train','validation','test')}
    for frame in data.values():
        require_columns(frame,['learner_id','text','proficiency_level'])
        if not frame['proficiency_level'].between(1,16).all():
            raise ValueError('Original proficiency level required 1..16')
    mid=cfg['backbone']; rev=cfg['revision'];tok=AutoTokenizer.from_pretrained(mid,revision=rev)
    model=AutoModelForSequenceClassification.from_pretrained(mid,revision=rev,num_labels=16)
    class Dataset(torch.utils.data.Dataset):
        def __init__(self,frame):
            # Lazy tokenization prevents materializing a ~840k x 512 token matrix.
            self.texts=frame['text'].fillna('').astype(str).tolist()
            self.labels=(frame['proficiency_level'].astype(int)-1).tolist()
        def __len__(self): return len(self.labels)
        def __getitem__(self,index):
            encoded=tok(self.texts[index],truncation=True,padding='max_length',max_length=cfg['max_length'])
            return {**{k:torch.tensor(v) for k,v in encoded.items()},
                    'labels':torch.tensor(self.labels[index])}
    def metrics(eval_pred):
        logits,labels=eval_pred
        pred=np.argmax(logits,axis=-1)
        return {'accuracy':accuracy_score(labels,pred),'weighted_f1':f1_score(labels,pred,average='weighted',zero_division=0)}
    out=str(Path(a.output_dir).resolve())
    args=TrainingArguments(output_dir=out,learning_rate=cfg['learning_rate'],
        per_device_train_batch_size=cfg['batch_size'],per_device_eval_batch_size=cfg['batch_size'],
        num_train_epochs=a.epochs or cfg['epochs'],weight_decay=cfg['weight_decay'],
        evaluation_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,
        metric_for_best_model='weighted_f1',greater_is_better=True,save_total_limit=2,
        report_to='none',seed=42)
    trainer=Trainer(model=model,args=args,train_dataset=Dataset(data['train']),
                    eval_dataset=Dataset(data['validation']),compute_metrics=metrics)
    trainer.train()
    result=trainer.evaluate(Dataset(data['test']),metric_key_prefix='test')
    trainer.save_model(out);tok.save_pretrained(out)
    save_json({'test':result,'model_id':mid,'revision':rev,'n_test':len(data['test'])},Path(out)/'final_metrics.json')
    print(result)
if __name__=='__main__': main()
