"""203 train / 40 development-validation / 60 sealed test split.  Test = the old 60 validation cases (untouched); the 40 development cases are drawn
from the old 243 training cases, stratified by slice thickness (thin <4 mm / thick) x vessel-burden tertile (union foreground fraction), fixed seed.
Reads cache/meta_all.json (prep_cluster.py) and the old split file; writes splits_v2.json and prints the stratum table."""
import json, os, sys
import numpy as np

ROOT = os.environ.get('SL_ROOT', os.path.expanduser('~/run/CSUqx/SpectraLink_v2'))
OLD = os.environ.get('SL_OLD_SPLIT', os.path.expanduser('~/run/CSUqx/SLinkNet/data/preprocess/splits.json'))
SEED, N_DEV = 2026, 40

meta = {m['id']: m for m in json.load(open(f'{ROOT}/cache/meta_all.json'))}
old = json.load(open(OLD)); tr_old, test = list(old['train']), list(old['val'])
assert len(set(tr_old) & set(test)) == 0 and len(tr_old) == 243 and len(test) == 60, (len(tr_old), len(test))
missing = [c for c in tr_old + test if c not in meta]; assert not missing, f'not cached: {missing[:5]}'
burden = np.array([meta[c]['frac_hv'] + meta[c]['frac_pv'] for c in tr_old]); q = np.quantile(burden, [1 / 3, 2 / 3])
strata = {}
for c, b in zip(tr_old, burden):
    key = ('thick' if meta[c]['z_mm'] >= 4 else 'thin', int((b > q[0]) + (b > q[1])))
    strata.setdefault(key, []).append(c)
rng = np.random.default_rng(SEED); dev = []
sizes = {k: len(v) for k, v in strata.items()}; tot = sum(sizes.values())
alloc = {k: int(np.floor(N_DEV * n / tot)) for k, n in sizes.items()}
for k in sorted(sizes, key=lambda k: -(N_DEV * sizes[k] / tot - alloc[k])):          # largest remainders first
    if sum(alloc.values()) >= N_DEV: break
    alloc[k] += 1
for k, v in sorted(strata.items()):
    pick = list(rng.choice(sorted(v), size=alloc[k], replace=False)); dev += pick
    print(f'stratum {k}: {len(v)} old-train cases -> {alloc[k]} dev')
dev = sorted(dev); train = sorted(c for c in tr_old if c not in set(dev)); test = sorted(test)
assert len(dev) == N_DEV and len(train) == 203 and not (set(dev) & set(train)) and not ((set(dev) | set(train)) & set(test))
out = dict(train=train, dev=dev, test=test, seed=SEED, note='test = old 60-case validation set, sealed; dev drawn from old training cases; strata = thickness x burden tertile')
json.dump(out, open(f'{ROOT}/splits_v2.json', 'w'), indent=1)
for name, ids in (('train', train), ('dev', dev), ('test', test)):
    z = np.array([meta[c]['z_mm'] for c in ids]); b = np.array([meta[c]['frac_hv'] + meta[c]['frac_pv'] for c in ids]) * 100
    print(f'{name:5s} n={len(ids):3d} thick(>=4mm)={int((z >= 4).sum()):3d} thin={int((z < 4).sum()):3d} vessel burden mean {b.mean():.2f}% median {np.median(b):.2f}%')
