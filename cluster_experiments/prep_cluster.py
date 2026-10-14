"""Cluster preprocessing (CPU only, resumable): raw MSD Task08 images + 3-class labels -> ROI cache identical to the local pilot (prep.py).
Image-only ROI (largest filled body component + 20 mm margin), resample to (z,y,x) = (2.5, 0.8, 0.8) mm, clip [-200, 300], z-score inside the ROI.
Writes <SL_ROOT>/cache/<id>.npz (img float16, lab uint8, c1/c2 = subsampled foreground coordinates) and <id>.json (meta). Atomic writes, skips finished cases.
Usage: python prep_cluster.py [--workers 16] [--limit N] [id ...]"""
import argparse, json, os, sys, time
import numpy as np, nibabel as nib
from multiprocessing import Pool
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

RAW = os.environ.get('SL_RAW', os.path.expanduser('~/run/CSUqx/DataBase/Task08_HepaticVessel'))
ROOT = os.environ.get('SL_ROOT', os.path.expanduser('~/run/CSUqx/SpectraLink_v2'))
TARGET = (2.5, 0.8, 0.8)
MARGIN_MM = 20.0


def body_bbox(hu, sp):
    small = hu[:, ::4, ::4] > -200
    small = ndi.binary_opening(small, iterations=1)
    small = np.stack([ndi.binary_fill_holes(s) for s in small])
    lab, n = ndi.label(small)
    if n == 0: return tuple(slice(0, s) for s in hu.shape)
    sizes = ndi.sum(small, lab, range(1, n + 1)); keep = lab == (1 + int(np.argmax(sizes)))
    idx = np.argwhere(keep); lo, hi = idx.min(0), idx.max(0) + 1
    lo = lo * np.array([1, 4, 4]); hi = hi * np.array([1, 4, 4])
    m = np.ceil(MARGIN_MM / np.array(sp)).astype(int)
    lo = np.maximum(lo - m, 0); hi = np.minimum(hi + m, hu.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def pool_cells(a):
    """half-resolution in-plane cell grid used by the candidate head: 2x2 max-pool over (y, x), z unchanged; odd trailing plane dropped."""
    z, y, x = a.shape
    return a[:, :y // 2 * 2, :x // 2 * 2].reshape(z, y // 2, 2, x // 2, 2).max(axis=(2, 4))


def coords(lab, c, cap=20000):
    pts = np.argwhere(lab == c)
    return pts[::max(1, len(pts) // cap)].astype(np.int16)


def prep_case(cid):
    out_npz, out_json = f'{ROOT}/cache/{cid}.npz', f'{ROOT}/cache/{cid}.json'
    if os.path.exists(out_npz) and os.path.exists(out_json): return json.load(open(out_json))
    t0 = time.time()
    img = nib.as_closest_canonical(nib.load(f'{RAW}/imagesTr/{cid}.nii.gz'))
    lab = nib.as_closest_canonical(nib.load(f'{RAW}/labelsTr/{cid}.nii.gz'))
    assert img.shape == lab.shape, (cid, img.shape, lab.shape)
    hu = np.asarray(img.dataobj).astype(np.float32).transpose(2, 1, 0)
    lb = np.asarray(lab.dataobj).astype(np.uint8).transpose(2, 1, 0)
    assert set(np.unique(lb).tolist()) <= {0, 1, 2}, (cid, np.unique(lb))
    sp = tuple(float(z) for z in img.header.get_zooms()[:3])[::-1]
    n_hv_raw, n_pv_raw = int((lb == 1).sum()), int((lb == 2).sum())
    sl = body_bbox(hu, sp)
    outside = int((lb > 0).sum() - (lb[sl] > 0).sum())
    hu, lb = hu[sl], lb[sl]
    fac = np.array(sp) / np.array(TARGET)
    hu_r = ndi.zoom(hu, fac, order=1, mode='nearest').astype(np.float32)
    lb_r = ndi.zoom(lb, fac, order=0, mode='nearest').astype(np.uint8)
    hu_r = np.clip(hu_r, -200, 300); mu, sd = float(hu_r.mean()), float(hu_r.std())
    hu_r = ((hu_r - mu) / (sd + 1e-6)).astype(np.float16)
    tmp = f'{ROOT}/cache/{cid}.tmp.npz'
    tg = {'tu': pool_cells(lb_r > 0), 'th': pool_cells(lb_r == 1), 'tp': pool_cells(lb_r == 2), 'tc': pool_cells(skeletonize(lb_r > 0))}   # cell-level targets for the recall budget
    np.savez_compressed(tmp, img=hu_r, lab=lb_r, c1=coords(lb_r, 1), c2=coords(lb_r, 2), cell_shape=np.array(tg['tu'].shape), **{k: np.packbits(v) for k, v in tg.items()})
    os.replace(tmp, out_npz)
    meta = dict(id=cid, orig_shape=list(lb.shape), roi_shape=list(hu_r.shape), spacing_orig=sp, z_mm=sp[0], raw_vox_hv=n_hv_raw, raw_vox_pv=n_pv_raw,
                label_vox_outside_roi=outside, frac_hv=float((lb_r == 1).mean()), frac_pv=float((lb_r == 2).mean()), seconds=round(time.time() - t0, 1))
    json.dump(meta, open(out_json + '.tmp', 'w')); os.replace(out_json + '.tmp', out_json)
    return meta


def work(cid):
    try: return prep_case(cid)
    except Exception as e: return dict(id=cid, error=repr(e))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('ids', nargs='*'); ap.add_argument('--workers', type=int, default=16); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(f'{ROOT}/cache', exist_ok=True)
    ids = a.ids or sorted(f[:-7] for f in os.listdir(f'{RAW}/imagesTr') if f.endswith('.nii.gz') and not f.startswith('._'))
    if a.limit: ids = ids[:a.limit]
    print(f'{len(ids)} cases, workers {a.workers}, raw {RAW}, out {ROOT}/cache', flush=True)
    bad = []
    with Pool(a.workers) as p:
        for i, m in enumerate(p.imap_unordered(work, ids), 1):
            if 'error' in m: bad.append(m); print('ERROR', m, flush=True)
            else: print(f"[{i}/{len(ids)}] {m['id']} roi {m['roi_shape']} z {m['z_mm']:.2f} mm HV {m['frac_hv']*100:.2f}% PV {m['frac_pv']*100:.2f}% outside_roi {m['label_vox_outside_roi']} ({m['seconds']}s)", flush=True)
    metas = [json.load(open(f'{ROOT}/cache/{c}.json')) for c in ids if os.path.exists(f'{ROOT}/cache/{c}.json')]
    json.dump(metas, open(f'{ROOT}/cache/meta_all.json', 'w'))
    print(f'done: {len(metas)}/{len(ids)} cached, {len(bad)} errors', flush=True)
    sys.exit(1 if bad else 0)
