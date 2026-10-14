"""CPU-only metrics from saved predictions: union / HV / PV Dice, clDice, surface ASD and HD95 (mm, grid (2.5,0.8,0.8)); writes <RUNS>/<tag>/results_<split>.json
in the same schema as the local results.json ({'cases': {id: {'HV','PV','union'}}}) plus cover_<split> recall tables merged into 'coverage'.
Empty predictions give NaN (counted as failures by analyze).  Usage: python metrics_cluster.py --tags A0_s17_full A6_s17_full --split dev [--workers 16]"""
import argparse, json, os
import numpy as np
from multiprocessing import Pool
import pilot as P

ap = argparse.ArgumentParser(); ap.add_argument('--tags', nargs='+', required=True); ap.add_argument('--split', default='dev'); ap.add_argument('--workers', type=int, default=16)
ap.add_argument('--splits', default=None)
a = ap.parse_args()


def one(args):
    tag, cid = args
    gt = np.load(f'{P.ROOT}/cache/{cid}.npz')['lab']; pred = np.load(f'{P.RUNS}/{tag}/pred_{a.split}/{cid}.npz')['pred']
    return tag, cid, {'HV': P.metrics_binary(pred == 1, gt == 1, P.SPACING), 'PV': P.metrics_binary(pred == 2, gt == 2, P.SPACING), 'union': P.metrics_binary(pred > 0, gt > 0, P.SPACING),
                      'pred_fg_frac': float((pred > 0).mean())}


if __name__ == '__main__':
    ids = json.load(open(a.splits or f'{P.ROOT}/splits_v2.json'))[a.split]
    jobs = []
    for t in a.tags:
        if os.path.exists(f'{P.RUNS}/{t}/results_{a.split}.json'): print('skip', t); continue
        miss = [c for c in ids if not os.path.exists(f'{P.RUNS}/{t}/pred_{a.split}/{c}.npz')]
        if miss: print(f'{t}: {len(miss)} predictions missing, skipping this tag'); continue
        jobs += [(t, c) for c in ids]
    res = {}
    with Pool(a.workers) as pool:
        for tag, cid, m in pool.imap_unordered(one, jobs, chunksize=1): res.setdefault(tag, {})[cid] = m
    for tag, cases in res.items():
        for cid in cases:
            cf = f'{P.RUNS}/{tag}/cover_{a.split}/{cid}.json'
            if os.path.exists(cf): cases[cid]['coverage'] = json.load(open(cf))
        json.dump(dict(tag=tag, split=a.split, n=len(cases), cases=cases), open(f'{P.RUNS}/{tag}/results_{a.split}.json', 'w'), indent=1)
        u = [c['union']['cldice'] for c in cases.values()]; print(f'{tag} {a.split}: n={len(cases)} union clDice mean {np.nanmean(u):.4f}, NaN {int(np.isnan(u).sum())}', flush=True)
