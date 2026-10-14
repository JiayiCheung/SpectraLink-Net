"""Paired patient-level analysis (protocol: bootstrap over cases, stratified by slice-thickness group, seeds averaged within case).
Usage: analyze_n20.py A6 A0   -> A6 minus A0.   Joins on case id, asserts identical case and seed sets, counts empty-prediction (NaN) cases."""
import json, os, sys, glob, numpy as np
R = '../runs'
Z = {}
for f in ('../../pilot_data/pilot20.json', '../../pilot_data/oldval_extra10.json'):
    if os.path.exists(f):
        for o in json.load(open(f)): Z[o['id']] = o['z']
def load(arm):
    out = {}
    for f in sorted(glob.glob(f'{R}/{arm}_s*_n20/results.json')):
        out[os.path.basename(os.path.dirname(f))] = json.load(open(f))['cases']
    return out
def per_case(arm, cls, k):
    runs = load(arm); assert runs, f'no results for {arm}'
    ids = sorted(next(iter(runs.values())))
    for name, r in runs.items(): assert sorted(r) == ids, f'case set differs in {name}'
    mat = np.array([[r[i][cls][k] for i in ids] for r in runs.values()], float)      # (seeds, cases)
    return ids, np.nanmean(mat, 0), list(runs), int(np.isnan(mat).any(0).sum())
def strat_boot(d, strata, n=10000, seed=0):
    rng = np.random.default_rng(seed); groups = [np.where(strata == s)[0] for s in np.unique(strata)]; m = np.empty(n)
    for b in range(n): m[b] = np.concatenate([rng.choice(d[g], len(g)) for g in groups]).mean()
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), int((d > 0).sum()), len(d)
def compare(a, b):
    ids, _, sa, _ = per_case(a, 'union', 'dice'); ids_b, _, sb, _ = per_case(b, 'union', 'dice')
    assert ids == ids_b, 'case sets differ between arms'
    strata = np.array([('thick' if Z.get(i, 5) >= 4 else 'thin') for i in ids])
    print(f'\n=== {a} - {b}  seeds {sa} vs {sb}; {len(ids)} cases ({(strata=="thin").sum()} thin <4 mm, {(strata=="thick").sum()} thick); strata-stratified bootstrap')
    for cls, k, lab in (('union', 'cldice', 'union clDice'), ('union', 'dice', 'union Dice'), ('HV', 'dice', 'HV Dice'), ('PV', 'dice', 'PV Dice'), ('union', 'asd', 'ASD mm'), ('union', 'hd95', 'HD95 mm')):
        _, x, _, nx = per_case(a, cls, k); _, y, _, ny = per_case(b, cls, k)
        ok = ~np.isnan(x - y); m, lo, hi, pos, n = strat_boot((x - y)[ok], strata[ok])
        print(f'{lab:13s} {np.nanmean(x):7.3f} vs {np.nanmean(y):7.3f} | diff {m:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}] | positive in {pos}/{n} | cases with NaN (empty/failed): {int((~ok).sum())}')
def coverage(tag):
    f = f'{R}/{tag}/coverage_volume.json'
    if not os.path.exists(f): return
    r = json.load(open(f)); cs = list(r); print(f'\n{tag}: whole-ROI recall (cell level) over {len(cs)} cases')
    for cls in ('union', 'HV', 'PV', 'centerline'):
        rows = [r[c]['head'][cls] for c in cs if cls in r[c]['head']]
        print(f'  {cls:10s} recall@5% {np.mean([x["recall_at"]["5%"] for x in rows])*100:5.1f} | @10% {np.mean([x["recall_at"]["10%"] for x in rows])*100:5.1f} | median budget for 90/95/98% recall ' + ' / '.join(f'{np.median([x["budget_for"][t] for x in rows])*100:4.1f}%' for t in ('90%', '95%', '98%')))
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) >= 2: compare(args[0], args[1])
    else:
        for t in sorted(glob.glob(f'{R}/*_n20/coverage_volume.json')): coverage(os.path.basename(os.path.dirname(t)))
