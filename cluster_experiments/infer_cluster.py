"""GPU inference only (resumable, per-case outputs).  For each case of a split: sliding-window prediction (window = P.PATCH, stride (16,48,48), Gaussian weights,
softmax averaged, full-ROI coverage asserted) -> <RUNS>/<tag>/pred_<split>/<id>.npz (uint8), and for arms with a candidate head the whole-ROI cell-recall budget
(cover_<split>/<id>.json, global ranking of all cells vs union/HV/PV/centreline cell targets stored in the cache).  Metrics (clDice, ASD, HD95) are computed
by metrics_cluster.py on a CPU-only job so that no GPU time is spent on skeletonisation / distance transforms.
Window starts along y/x are rounded to even so the half-resolution cell grid stays aligned (z exact).
Usage: python infer_cluster.py --tag A6_s17_full --split dev [--arm A6] [--max_seconds N]"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
import pilot as P

ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--arm', default=None); ap.add_argument('--split', default='dev')
ap.add_argument('--splits', default=None); ap.add_argument('--max_seconds', type=float, default=0); ap.add_argument('--limit_cases', type=int, default=0)
a = ap.parse_args()
arm = a.arm or a.tag.split('_')[0]
out = f'{P.RUNS}/{a.tag}'; os.makedirs(f'{out}/pred_{a.split}', exist_ok=True); os.makedirs(f'{out}/cover_{a.split}', exist_ok=True)
ids = json.load(open(a.splits or f'{P.ROOT}/splits_v2.json'))[a.split]
if a.limit_cases: ids = ids[:a.limit_cases]
dev = P.dev
net = P.Net(arm).to(dev); net.load_state_dict(torch.load(f'{out}/model.pt', map_location=dev)); net.eval()
BUD, TGT = (0.01, 0.02, 0.05, 0.10, 0.20), (0.90, 0.95, 0.98)


@torch.no_grad()
def infer_case(vol, stride=(16, 48, 48)):
    Z, Y, X = vol.shape
    pad = [max(0, w - s) for w, s in zip(P.PATCH, vol.shape)]; pad[1] += (Y + pad[1]) % 2; pad[2] += (X + pad[2]) % 2
    v = np.pad(vol, [(0, p) for p in pad], mode='constant', constant_values=float(vol.min())); Zp, Yp, Xp = v.shape
    vt = torch.from_numpy(v.astype(np.float32)).to(dev)
    acc = torch.zeros(3, Zp, Yp, Xp, device=dev); cnt = torch.zeros(Zp, Yp, Xp, device=dev)
    cacc = torch.zeros(Zp, Yp // 2, Xp // 2, device=dev); ccnt = torch.zeros_like(cacc)
    sig = [w / 8 for w in P.PATCH]; grid = np.meshgrid(*[np.arange(w) - (w - 1) / 2 for w in P.PATCH], indexing='ij')
    gw = np.exp(-sum((g / s) ** 2 for g, s in zip(grid, sig)) / 2); gw = torch.from_numpy(np.maximum(gw / gw.max(), 1e-3)).float().to(dev)
    gz, gy, gx = [np.arange(n) - (n - 1) / 2 for n in (P.PATCH[0], P.PATCH[1] // 2, P.PATCH[2] // 2)]
    gc = np.exp(-(gz[:, None, None] ** 2 / (2 * (P.PATCH[0] / 8) ** 2) + gy[None, :, None] ** 2 / (2 * (P.PATCH[1] / 16) ** 2) + gx[None, None, :] ** 2 / (2 * (P.PATCH[2] / 16) ** 2)))
    gc = torch.from_numpy(np.maximum(gc / gc.max(), 1e-3)).float().to(dev)
    def starts(L, w, st, even):
        s = list(range(0, max(1, L - w + 1), st)); s = s if (s[-1] == L - w or L <= w) else s + [L - w]
        return [t - (t % 2) for t in s] if even else s
    wins = [(i, j, k) for i in starts(Zp, P.PATCH[0], stride[0], False) for j in starts(Yp, P.PATCH[1], stride[1], True) for k in starts(Xp, P.PATCH[2], stride[2], True)]
    pz, py, px = P.PATCH
    for s in range(0, len(wins), 4):
        chunk = wins[s:s + 4]; x = torch.stack([vt[i:i + pz, j:j + py, k:k + px] for i, j, k in chunk]).unsqueeze(1)
        with torch.autocast('cuda', dtype=torch.float16): lg, cl = net(x)
        pr = F.softmax(lg.float(), 1)
        for n, (i, j, k) in enumerate(chunk):
            acc[:, i:i + pz, j:j + py, k:k + px] += pr[n] * gw; cnt[i:i + pz, j:j + py, k:k + px] += gw
            if cl is not None:
                cacc[i:i + pz, j // 2:j // 2 + py // 2, k // 2:k // 2 + px // 2] += cl[n, 0].float() * gc; ccnt[i:i + pz, j // 2:j // 2 + py // 2, k // 2:k // 2 + px // 2] += gc
    assert float(cnt.min()) > 0, 'uncovered voxels'
    pred = (acc / cnt).argmax(0)[:Z, :Y, :X].cpu().numpy().astype(np.uint8)
    score = None
    if net.ref is not None or net.early is not None:
        assert float(ccnt[:Z, :Y // 2, :X // 2].min()) > 0, 'uncovered cells'
        score = (cacc / ccnt.clamp(min=1e-6))[:Z, :Y // 2, :X // 2].cpu().numpy()
    return pred, score, len(wins)


def recall(score, tgts):
    order = np.argsort(-score.ravel(), kind='stable'); res = {}
    for name, t in tgts.items():
        tf = t.ravel()[order].astype(np.float64); n = tf.sum()
        if n == 0: continue
        cum = np.cumsum(tf) / n
        res[name] = {'recall_at': {f'{int(b * 100)}%': float(cum[max(0, int(round(b * len(tf))) - 1)]) for b in BUD},
                     'budget_for': {f'{int(t_ * 100)}%': float((np.searchsorted(cum, t_) + 1) / len(tf)) for t_ in TGT}}
    return res


t_start = time.time(); done = 0
for cid in ids:
    po, co = f'{out}/pred_{a.split}/{cid}.npz', f'{out}/cover_{a.split}/{cid}.json'
    if os.path.exists(po) and (os.path.exists(co) or (net.ref is None and net.early is None)): continue
    if a.max_seconds and time.time() - t_start > a.max_seconds: print('time guard: rerun to continue', flush=True); sys.exit(3)
    t0 = time.time(); z = np.load(f'{P.ROOT}/cache/{cid}.npz'); vol = z['img'].astype(np.float32)
    torch.manual_seed(0); pred, score, nw = infer_case(vol)
    np.savez_compressed(po + '.tmp.npz', pred=pred); os.replace(po + '.tmp.npz', po)
    if score is not None:
        cs = tuple(int(v) for v in z['cell_shape']); tg = {k: np.unpackbits(z[t])[:int(np.prod(cs))].reshape(cs).astype(bool) for k, t in (('union', 'tu'), ('HV', 'th'), ('PV', 'tp'), ('centerline', 'tc'))}
        assert score.shape == cs, (score.shape, cs)
        json.dump(dict(n_cells=int(score.size), head=recall(score, tg), windows=nw), open(co + '.tmp', 'w')); os.replace(co + '.tmp', co)
    done += 1; print(f'{a.tag} {a.split} {cid} {nw} windows {time.time()-t0:.0f}s', flush=True)
print(f'done {done} new cases in {time.time()-t_start:.0f}s', flush=True)
