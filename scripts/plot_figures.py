"""Rebuild figures 4-8 from provided summary means. No fake error bars."""
import argparse
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

p=argparse.ArgumentParser();p.add_argument('--input',default='data/public_summary/figures_4_8.csv');p.add_argument('--output-dir',default='results/figures');a=p.parse_args()
df=pd.read_csv(ROOT/a.input);out=ROOT/a.output_dir;out.mkdir(parents=True,exist_ok=True)
for (figure,metric),g in df.groupby(['figure','indicator'],sort=True):
    prepost = set(g.time)=={'pre','post'}
    fig,ax=plt.subplots(figsize=(6,4))
    if prepost:
        wide=g.pivot(index='group',columns='time',values='value').reindex(['experimental','control'])
        x=np.arange(2);w=.35
        ax.bar(x-w/2,wide['pre'],width=w,label='Pre')
        ax.bar(x+w/2,wide['post'],width=w,label='Post')
        ax.set_xticks(x,['Experimental','Control']);ax.legend()
    else:
        g=g.set_index('group').reindex(['experimental','control'])
        ax.bar(['Experimental','Control'],g.value)
    ax.set_ylabel(g.unit.iloc[0]);ax.set_title(f'Figure {figure}: {metric}')
    ax.text(.01,-.23,'Source: provided workbook summary means; SD/CI unavailable.',transform=ax.transAxes,fontsize=8)
    fig.subplots_adjust(bottom=.28)
    fig.savefig(out/f'figure_{figure}_{metric}.png',dpi=180)
    plt.close(fig)
print(f'Generated summary-only figures under {out}')
