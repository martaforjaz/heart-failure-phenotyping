"""Make publication-style panels from the UCI survival benchmark."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

data = json.loads(Path('uci-results.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.8), layout='constrained')
palette = ['#25598A', '#468278', '#B47A4B']
for (name, metrics), color in zip(data['methods'].items(), palette):
    score = metrics['stability']
    axes[0].plot(list(map(int, score)), list(score.values()), 'o-', color=color, label=name)
axes[0].set(xticks=[2, 3, 4, 5], ylim=(0, 1), xlabel='Number of clusters',
            ylabel='Mean pairwise ARI', title='A   Clustering stability')
axes[0].legend(frameon=False)
for ax, panel, (name, metrics) in zip(axes[1:], 'BC', data['methods'].items()):
    for row, color in zip(metrics['km'], palette):
        time, estimate, low, high = np.asarray(row['curve']).T
        ax.step(time, estimate, where='post', color=color, label=f"Group {row['group']} (n={row['n']})")
        ax.fill_between(time, low, high, step='post', color=color, alpha=.13)
    ax.set(xlabel='Follow-up (days)', title=f'{panel}   {name}', ylim=(0, 1.04),
           xticks=[0, 90, 180, 270])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, fontsize=8, loc='lower left')
axes[1].set_ylabel('Estimated survival')
for extension in ('png', 'pdf'):
    fig.savefig(f'figures/uci-survival.{extension}', dpi=300)
