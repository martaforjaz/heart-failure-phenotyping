"""Reproduce the UCI heart failure benchmark from the official archive."""
import argparse
import json
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from analyze import vae

CONTINUOUS = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',
              'serum_creatinine', 'serum_sodium']
BINARY = ['anaemia', 'diabetes', 'high_blood_pressure', 'smoking']
SEEDS = (4, 15, 26, 37, 48)
CLUSTERS = 3


def load(archive):
    with ZipFile(archive) as source:
        with source.open('heart_failure_clinical_records_dataset.csv') as stream:
            data = pd.read_csv(stream)
    cols = [*CONTINUOUS, *BINARY, 'sex', 'time', 'DEATH_EVENT']
    if len(data) != 299 or data[cols].isna().any().any() or (data.time <= 0).any():
        raise ValueError('Unexpected UCI release or missing observations.')
    if not data[[*BINARY, 'sex', 'DEATH_EVENT']].isin([0, 1]).all().all():
        raise ValueError('Expected binary fields.')
    return data


def features(train, test):
    def continuous(part):
        x = part[CONTINUOUS].copy()
        for col in ('creatinine_phosphokinase', 'platelets', 'serum_creatinine'):
            x[col] = np.log1p(x[col])
        return x

    train_values, test_values = continuous(train), continuous(test)
    scale = StandardScaler().fit(train_values)
    return tuple(np.column_stack([scale.transform(values), part[BINARY].to_numpy()])
                 for values, part in ((train_values, train), (test_values, test)))


def survival(test, labels):
    follow = test[['time', 'DEATH_EVENT']].reset_index(drop=True).assign(group=labels+1)
    curves = []
    for group, rows in follow.groupby('group'):
        km = KaplanMeierFitter().fit(rows.time, rows.DEATH_EVENT)
        bounds = km.confidence_interval_.to_numpy()
        curves.append({'group': int(group), 'n': len(rows), 'deaths': int(rows.DEATH_EVENT.sum()),
                       'curve': np.column_stack([km.timeline, km.survival_function_.to_numpy(),
                                                bounds]).tolist(),
                       'at_risk': [int((rows.time >= day).sum()) for day in (0, 90, 180, 270)]})
    cox = None
    if len(curves) == CLUSTERS and all(row['deaths'] >= 5 for row in curves):
        design = pd.get_dummies(follow, columns=['group'], drop_first=True, dtype=int)
        model = CoxPHFitter().fit(design, duration_col='time', event_col='DEATH_EVENT')
        ph = proportional_hazard_test(model, design, time_transform='rank').summary
        cox = [{'comparison': f'{key.removeprefix("group_")} vs 1',
                'hr': float(row['exp(coef)']),
                'low': float(row['exp(coef) lower 95%']),
                'high': float(row['exp(coef) upper 95%']),
                'ph_test_p': float(ph.loc[key, 'p'])}
               for key, row in model.summary.iterrows()]
    return {'km': curves, 'cox': cox}


def analyze(data):
    train, test = train_test_split(data, test_size=.4, random_state=42)
    x, y = features(train, test)
    transforms = {'Clinical features': [(x, y)] * len(SEEDS),
                  'VAE features': [vae(x, y, seed) for seed in SEEDS]}
    result = {'dataset': 'UCI Heart Failure Clinical Records', 'n': len(data),
              'train_n': len(train), 'test_n': len(test),
              'deaths': int(data.DEATH_EVENT.sum()), 'methods': {}}
    younger = data[data.sex.eq(0) & data.age.lt(55)]
    result['women_under_55'] = {'n': len(younger), 'deaths': int(younger.DEATH_EVENT.sum())}
    for method, runs in transforms.items():
        labels_by_k = {}
        stability = {}
        for k in range(2, 6):
            labels = [KMeans(n_clusters=k, n_init=10, random_state=seed).fit(a).predict(b)
                      for seed, (a, b) in zip(SEEDS, runs)]
            labels_by_k[k] = labels[0]
            stability[k] = float(np.mean([adjusted_rand_score(a, b)
                                          for a, b in combinations(labels, 2)]))
        labels = labels_by_k[CLUSTERS]
        result['methods'][method] = {'stability': stability, **survival(test, labels),
                                     'profiles': test.assign(group=labels+1).groupby('group')[
                                         [*CONTINUOUS, *BINARY]].median().to_dict(orient='index')}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = analyze(load(args.archive))
    Path('uci-results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print({k:v for k,v in result.items() if k != 'methods'})
    for method, metrics in result['methods'].items():
        print(method, metrics['stability'], [(v['n'],v['deaths']) for v in metrics['km']], metrics['cox'])
