"""Cluster training, ONE ARM PER PROCESS (several arms share one GPU by running several processes).  Uses pilot.py (same directory) for Net / losses.
- DataLoader workers prefetch patches (the single-process sampler costs ~49 ms per patch, it would starve an A100).
- bf16 autocast, cudnn.benchmark, AdamW lr 1e-3 cosine with 5% warm-up, clip 1.0, loss = 0.7 Dice + 0.3 CE + aux_w * candidate loss (as pilot.run).
- Step-based atomic checkpoints (resume after kill / time limit), NaN stop (exit 2), time guard (exit 3 = resume me), no model selection: final weights only.
Usage: python train_cluster.py --arm A6 --seed 17 --tag A6_s17_full --split train --steps 12000 [--workers 4] [--max_seconds 40000] [--limit_cases N]"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
import pilot as P

ap = argparse.ArgumentParser()
ap.add_argument('--arm', required=True); ap.add_argument('--seed', type=int, default=17); ap.add_argument('--tag', required=True)
ap.add_argument('--splits', default=None); ap.add_argument('--split', default='train'); ap.add_argument('--limit_cases', type=int, default=0)
ap.add_argument('--steps', type=int, default=12000); ap.add_argument('--bs', type=int, default=2); ap.add_argument('--workers', type=int, default=4)
ap.add_argument('--ckpt_every', type=int, default=500); ap.add_argument('--max_seconds', type=float, default=0); ap.add_argument('--log_every', type=int, default=50)
a = ap.parse_args()
dev = P.dev
torch.backends.cudnn.benchmark = True
out = f'{P.RUNS}/{a.tag}'; os.makedirs(out, exist_ok=True); ckpt = f'{out}/ckpt.pt'
logf = open(f'{out}/train_log.txt', 'a', encoding='utf-8')
def L(*x):
    s = ' '.join(str(t) for t in x); print(s, flush=True); logf.write(s + '\n'); logf.flush()

if os.path.exists(f'{out}/done.json'): L('already done, exiting'); sys.exit(0)
ids = json.load(open(a.splits or f'{P.ROOT}/splits_v2.json'))[a.split]
if a.limit_cases: ids = ids[:a.limit_cases]
t_load = time.time(); data = P.Data(ids); L(f'loaded {len(ids)} cases in {time.time()-t_load:.0f}s')


class PatchStream(torch.utils.data.IterableDataset):
    """endless stream of patches: 35% HV-centred, 35% PV-centred, 30% random (P.Data.sample); per-worker RNG, offset by the resume step."""
    def __init__(s, data, ids, seed, offset): s.data, s.ids, s.seed, s.offset = data, ids, seed, offset
    def __iter__(s):
        wi = torch.utils.data.get_worker_info(); wid = wi.id if wi else 0
        rng = np.random.default_rng([s.seed, wid, s.offset])
        while True:
            x, y = s.data.sample(rng, s.ids[rng.integers(len(s.ids))])
            yield torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y.astype(np.int64))


torch.manual_seed(a.seed)
net = P.Net(a.arm).to(dev); npar = sum(p.numel() for p in net.parameters())
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
step, elapsed = 0, 0.0
if os.path.exists(ckpt):
    c = torch.load(ckpt, map_location=dev); net.load_state_dict(c['net']); opt.load_state_dict(c['opt']); step, elapsed = c['step'], c['elapsed']
    L(f'resumed from step {step} (elapsed {elapsed:.0f}s)')
L(f'arm={a.arm} seed={a.seed} params={npar/1e6:.3f}M steps={a.steps} bs={a.bs} workers={a.workers} cases={len(ids)} split={a.split}')
warm = max(1, int(0.05 * a.steps))
lr_at = lambda t: 1e-3 * (t + 1) / warm if t < warm else 1e-6 + 0.5 * (1e-3 - 1e-6) * (1 + math.cos(math.pi * (t - warm) / max(1, a.steps - warm)))
loader = torch.utils.data.DataLoader(PatchStream(data, ids, a.seed, step), batch_size=a.bs, num_workers=a.workers, pin_memory=True, prefetch_factor=(4 if a.workers > 0 else None), persistent_workers=a.workers > 0)
it = iter(loader)
def save(final=False):
    tmp = ckpt + '.tmp'; torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), step=step, elapsed=elapsed), tmp); os.replace(tmp, ckpt)

t0 = time.time(); t_log = t0; wait = 0.0; acc = []; net.train()
while step < a.steps:
    tw = time.time(); x, y = next(it); wait += time.time() - tw
    x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
    x = x * (1 + 0.1 * (torch.rand(x.shape[0], 1, 1, 1, 1, device=dev) * 2 - 1)) + 0.1 * (torch.rand(x.shape[0], 1, 1, 1, 1, device=dev) * 2 - 1)
    for g in opt.param_groups: g['lr'] = lr_at(step)
    net.gt_cells = F.max_pool3d((y > 0).float().unsqueeze(1), (1, 2, 2))
    with torch.autocast('cuda', dtype=torch.bfloat16): lg, cl = net(x)
    loss, ce = P.seg_loss(lg, y); aux = P.cand_loss(cl, y, tversky=(a.arm != 'A7')) if cl is not None else torch.zeros((), device=dev)
    if net.cl2 is not None: aux = aux + P.cand_loss(net.cl2, y, tversky=True)
    tot = loss + net.aux_w * aux
    if net.ds_out is not None:
        frac = F.avg_pool3d(F.one_hot(y, 3).permute(0, 4, 1, 2, 3).float(), (1, 2, 2))
        tot = tot + 0.2 * (-(frac * F.log_softmax(net.ds_out[0].float(), 1)).sum(1).mean())
    if not torch.isfinite(tot): L(f'NaN/inf loss at step {step}, stopping (checkpoint kept at last good step)'); sys.exit(2)
    opt.zero_grad(set_to_none=True); tot.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); step += 1
    acc.append((float(loss), float(aux)))
    if step % a.log_every == 0:
        now = time.time(); sps = a.log_every / (now - t_log)
        L(f'step {step}/{a.steps} seg {np.mean([e[0] for e in acc]):.4f} aux {np.mean([e[1] for e in acc]):.4f} lr {lr_at(step):.2e} {sps:.2f} it/s wait {wait/a.log_every*1000:.0f} ms/it mem {torch.cuda.max_memory_allocated()/2**30:.2f} GB')
        t_log, wait, acc = now, 0.0, []
    if step % a.ckpt_every == 0 or step == a.steps:
        elapsed += time.time() - t0; t0 = time.time(); save()
    if a.max_seconds and (elapsed + time.time() - t0) > a.max_seconds and step < a.steps:
        elapsed += time.time() - t0; save(); L(f'time guard at step {step}: checkpoint saved, resubmit to continue'); sys.exit(3)
elapsed += time.time() - t0
torch.save(net.state_dict(), f'{out}/model.pt')
json.dump(dict(arm=a.arm, seed=a.seed, steps=step, train_seconds=elapsed, params_M=npar / 1e6, cases=len(ids), split=a.split), open(f'{out}/done.json', 'w'))
L(f'done: {step} steps in {elapsed:.0f}s')
