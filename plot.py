"""Render the synthetic benchmark, using figures4papers as the style reference."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

data = json.loads(Path('synthetic-results.json').read_text(encoding='utf-8'))
assert data['data'] == 'synthetic'
plt.rcParams.update({'font.family':'sans-serif','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'figure.dpi':150})
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout='constrained')
palette = ['#25598A','#468278','#B47A4B']
ax = axes[0]
for (name, values), color in zip(data['stability'].items(), palette):
    ax.plot([int(k) for k in values], list(values.values()), 'o-', color=color, label=name)
ax.set(xticks=[2,3,4,5], xlabel='Cluster count', ylabel='Mean pairwise ARI', ylim=(-.05,1.08),
       title='A   Clustering stability')
ax.legend(frameon=False, fontsize=9)
for panel, ax, (name, rows) in zip('BC', axes[1:], data['outcomes'].items()):
    for cluster, color in enumerate(palette, 1):
        records = [r for r in rows if r['cluster']==cluster and r['rate'] is not None]
        points = [r['rate'] for r in records]
        ax.errorbar([r['days'] for r in records], points,
                    yerr=[[max(0, r['rate']-r['low']) for r in records],
                          [max(0, r['high']-r['rate']) for r in records]],
                    fmt='o-', capsize=3, label=f'Group {cluster}', color=color)
    ax.set(xticks=[28,90,180], xlabel='Days after index discharge', ylim=(0,1),
           title=f'{panel}   {name}')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
axes[1].set_ylabel('Readmitted by window')
axes[2].legend(frameon=False, fontsize=9)
fig.suptitle('Synthetic heart failure benchmark', fontweight='bold')
Path('figures').mkdir(exist_ok=True)
for extension in ('png','pdf'):
    fig.savefig(f'figures/synthetic-benchmark.{extension}',dpi=300)
