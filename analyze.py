"""Explore clustering stability and held-out fixed-window readmission in a patient cohort."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

FEATURES = ('systolic_bp', 'pulse', 'bmi', 'nyha')
OUTCOMES = ('readmit_28', 'readmit_90', 'readmit_180')


def synthetic():
    rng = np.random.default_rng(42)
    group = rng.integers(0, 3, 750)
    values = rng.normal(0, [12, 10, 4, .35], (750, 4)) + np.array(
        [[112, 72, 27, 2], [153, 78, 32, 2], [117, 105, 25, 3.4]])[group]
    data = pd.DataFrame(values, columns=FEATURES)
    data['patient_id'] = np.arange(len(data))
    data['age'] = rng.integers(25, 91, len(data))
    data['sex'] = rng.choice(['female', 'male'], len(data))
    days = rng.exponential(np.array([550, 280, 140])[group])
    for label, cutoff in zip(OUTCOMES, (28, 90, 180)):
        data[label] = (days <= cutoff).astype(int)
    return data


def vae(train, test, seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(train.shape[1], 24), nn.SiLU(), nn.Linear(24, 6))
    decoder = nn.Sequential(nn.Linear(3, 24), nn.SiLU(), nn.Linear(24, train.shape[1]))
    optimizer = torch.optim.Adam([*model.parameters(), *decoder.parameters()], lr=.006)
    samples = torch.tensor(train, dtype=torch.float32)
    for _ in range(80):
        for batch in samples[torch.randperm(len(samples))].split(96):
            mu, logvar = model(batch).chunk(2, dim=1)
            logvar = logvar.clamp(-8, 6)
            z = mu + torch.randn_like(mu) * (logvar * .5).exp()
            loss = (decoder(z) - batch).square().mean() + .05 * (
                logvar.exp() + mu.square() - 1 - logvar).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        return tuple(model(torch.tensor(x, dtype=torch.float32))[:, :3].numpy() for x in (train, test))


def wilson(events, n):
    if not n:
        return [None, None, None]
    p = events / n
    z2 = 1.96**2
    center = (p + z2 / (2*n)) / (1 + z2/n)
    half = 1.96 * np.sqrt(p*(1-p)/n + z2/(4*n*n))/(1+z2/n)
    return [p, max(0., center-half), min(1., center+half)]


def run(data, kind):
    required = ['patient_id', 'age', 'sex', *FEATURES, *OUTCOMES]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f'Missing normalized fields: {missing}')
    if data.patient_id.isna().any() or data.patient_id.duplicated().any():
        raise ValueError('Expected one index hospitalization per unique patient.')
    for col in OUTCOMES:
        if not data[col].dropna().isin([0, 1]).all():
            raise ValueError(f'{col} must contain 0, 1 or missing.')
    train, test = train_test_split(data, test_size=.30, random_state=42)
    imputer = SimpleImputer(strategy='median').fit(train[list(FEATURES)])
    x, y = [imputer.transform(part[list(FEATURES)]) for part in (train, test)]
    scaler = StandardScaler().fit(x)
    x, y = scaler.transform(x), scaler.transform(y)
    vae_runs = [vae(x, y, seed) for seed in (4, 15, 26)]
    embeddings = {'Clinical features': [(x, y)] * 3, 'VAE features': vae_runs}
    output = {'data':kind, 'train':len(train), 'test':len(test), 'stability':{}, 'outcomes':{}, 'profiles':{}}
    for name, variants in embeddings.items():
        predictions = {}
        for k in range(2, 6):
            predictions[k] = [KMeans(n_clusters=k, n_init=10, random_state=seed).fit(a).predict(b)
                              for seed, (a,b) in zip((4,15,26), variants)]
        stability = {k: float(np.mean([adjusted_rand_score(a, b)
                     for i, a in enumerate(labels) for b in labels[i+1:]]))
                     for k, labels in predictions.items()}
        output['stability'][name] = stability
        # Both approaches use the same prespecified k for the outcome comparison.
        labels = predictions[3][0]
        outcomes, profiles = [], []
        for cluster in range(3):
            group = test.iloc[labels == cluster]
            profiles.append({'cluster':cluster+1,'n':len(group),
                             'means':group[list(FEATURES)].mean().to_dict()})
            for cutoff, col in zip((28,90,180),OUTCOMES):
                known = group[col].dropna()
                point, low, high = wilson(int(known.sum()),len(known))
                outcomes.append({'cluster':cluster+1,'days':cutoff,'n':len(known),
                                 'readmitted':int(known.sum()),'rate':point,'low':low,'high':high})
        output['outcomes'][name] = outcomes
        output['profiles'][name] = profiles
    younger = test[test.sex.astype(str).str.lower().eq('female') & (test.age < 55)]
    output['exploratory_younger_women'] = {'threshold_years':55, 'n':len(younger),
                                         'known_180':int(younger.readmit_180.notna().sum()),
                                         'readmissions_180':int(younger.readmit_180.sum())}
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Privately prepared normalized cohort CSV. Omit for synthetic data.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.input and not args.output:
        parser.error('Provide a private output path for a clinical run.')
    torch.set_num_threads(1)
    result = run(pd.read_csv(args.input) if args.input else synthetic(), 'clinical' if args.input else 'synthetic')
    path = args.output or Path('synthetic-results.json')
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
