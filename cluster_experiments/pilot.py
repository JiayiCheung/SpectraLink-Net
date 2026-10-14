"""Local pilot: spatial-only (A0) vs supervised sparse refinement (A2) vs random-selection control (A4) vs dense queries (A5).
Axes are (z, y, x).  3 classes: 0 background, 1 HV, 2 PV.  Nothing here touches the cluster."""
import argparse, json, math, os, re, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

ROOT = os.environ.get('SL_ROOT', 'G:/Article/SpectraLink_Review/pilot_data')
RUNS = os.environ.get('SL_RUNS', 'G:/Article/SpectraLink_Review/work/runs')
PATCH = (32, 96, 96)
SPACING = (2.5, 0.8, 0.8)
dev = torch.device('cuda')


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, ids):
        self.ids, self.vol, self.lab, self.coords = ids, {}, {}, {}
        from concurrent.futures import ThreadPoolExecutor
        def load(i):
            z = np.load(f'{ROOT}/cache/{i}.npz'); img, lab = z['img'], z['lab']
            if 'c1' in z.files: co = {1: z['c1'].astype(np.int64), 2: z['c2'].astype(np.int64)}       # precomputed by prep_cluster.py
            else: co = {c: np.argwhere(lab == c)[::max(1, int((lab == c).sum() // 20000))] for c in (1, 2)}
            return i, img, lab, co
        with ThreadPoolExecutor(int(os.environ.get('SL_LOAD_THREADS', '8'))) as ex:                       # zlib releases the GIL
            for i, img, lab, co in ex.map(load, ids): self.vol[i], self.lab[i], self.coords[i] = img, lab, co

    def extract(self, cid, center):
        img, lab = self.vol[cid], self.lab[cid]; size = np.array(PATCH)
        start = np.array(center) - size // 2
        lo, hi = np.maximum(start, 0), np.minimum(start + size, img.shape); d0 = lo - start; d1 = d0 + (hi - lo)
        oi = np.full(PATCH, float(img.min()), np.float32); ol = np.zeros(PATCH, np.uint8)
        sl_s = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi)); sl_d = tuple(slice(int(a), int(b)) for a, b in zip(d0, d1))
        oi[sl_d] = img[sl_s]; ol[sl_d] = lab[sl_s]
        return oi, ol

    def sample(self, rng, cid):
        r = rng.random(); sh = self.vol[cid].shape
        if r < 0.70:
            c = 1 if r < 0.35 else 2; pts = self.coords[cid][c]
            center = pts[rng.integers(len(pts))] if len(pts) else [rng.integers(s) for s in sh]
        else:
            center = [rng.integers(s) for s in sh]
        return self.extract(cid, center)


# ----------------------------------------------------------------------------- model
class SepPath(nn.Module):
    def __init__(s, cin, cout, k, kz):
        super().__init__()
        s.dw = nn.Conv3d(cin, cin, (kz, k, k), padding=(kz // 2, k // 2, k // 2), groups=cin, bias=False)
        s.pw = nn.Conv3d(cin, cout, 1, bias=False)
    def forward(s, x): return s.pw(s.dw(x))


def masked_in(norm, y, m):
    """instance norm whose statistics use only active sites (m: (B,1,Z,Y,X) in {0,1}); equals InstanceNorm3d when m == 1."""
    y = y.float(); n = m.sum((2, 3, 4), keepdim=True).clamp(min=1)
    mu = (y * m).sum((2, 3, 4), keepdim=True) / n; var = (((y - mu) ** 2) * m).sum((2, 3, 4), keepdim=True) / n
    return (y - mu) / torch.sqrt(var + norm.eps) * norm.weight.view(1, -1, 1, 1, 1) + norm.bias.view(1, -1, 1, 1, 1)


class MSBlock(nn.Module):
    """three separable paths with in-plane kernels 3/5/7 (z kernel kz), channel split 1/2 : 1/4 : 1/4, residual."""
    def __init__(s, cin, cout, kz):
        super().__init__()
        c1, c2 = cout // 2, cout // 4; c3 = cout - c1 - c2
        s.p1, s.p2, s.p3 = SepPath(cin, c1, 3, kz), SepPath(cin, c2, 5, kz), SepPath(cin, c3, 7, kz)
        s.norm = nn.InstanceNorm3d(cout, affine=True); s.fuse = nn.Conv3d(cout, cout, 1)
        s.short = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()
    def forward(s, x, m=None):
        y = torch.cat([s.p1(x), s.p2(x), s.p3(x)], 1)
        if m is None: return F.relu(s.fuse(F.relu(s.norm(y))) + s.short(x))
        out = F.relu(s.fuse(F.relu(masked_in(s.norm, y, m))) + s.short(x))     # simulated sparse execution: active sites only
        return out * m


class SparseRefine(nn.Module):
    """candidate head + (top-k | random | dense) query selection + cross-attention to pooled context tokens +
    score-weighted residual on selected positions only:  x_i' = x_i + gamma * p_i * Delta_i  (i in S), x_i' = x_i otherwise."""
    def __init__(s, c, mode, k_frac=0.10, heads=4, pool=(4, 6, 6), c_ctx=None, gamma0=0.1):
        super().__init__()
        s.mode, s.k_frac, s.heads, s.pool = mode, k_frac, heads, pool
        s.cand = nn.Conv3d(c, 1, 1)
        s.ctx_proj = nn.Linear(c_ctx, c) if c_ctx else None
        s.nq, s.nkv = nn.LayerNorm(c), nn.LayerNorm(c)
        s.q, s.k, s.v, s.o = (nn.Linear(c, c) for _ in range(4))
        s.gamma = nn.Parameter(torch.tensor(float(gamma0)))
    def forward(s, x, ctx_src=None):
        B, C, Z, Y, X = x.shape
        cl = s.cand(x)
        if s.mode == 'none': return x, cl                                               # (B,1,Z,Y,X) candidate logits
        flat = x.flatten(2).transpose(1, 2); N = flat.shape[1]       # (B,N,C)
        sc = cl.flatten(2).squeeze(1)                                # (B,N)
        K = N if s.mode == 'dense' else max(1, int(round(s.k_frac * N)))
        if s.mode == 'dense': idx = torch.arange(N, device=x.device).expand(B, N)
        elif s.mode == 'topk': idx = sc.detach().float().topk(K, dim=1).indices
        else: idx = torch.rand(B, N, device=x.device).topk(K, dim=1).indices          # random ranking
        src = x if ctx_src is None else ctx_src
        ctx = F.adaptive_avg_pool3d(src, s.pool).flatten(2).transpose(1, 2)                # (B,T,.) compact context tokens
        if s.ctx_proj is not None: ctx = s.ctx_proj(ctx)
        qs = torch.gather(flat, 1, idx.unsqueeze(-1).expand(-1, -1, C))
        q = s.q(s.nq(qs)); k = s.k(s.nkv(ctx)); v = s.v(s.nkv(ctx))
        sp = lambda t: t.view(B, t.shape[1], s.heads, C // s.heads).transpose(1, 2)
        a = F.scaled_dot_product_attention(sp(q), sp(k), sp(v)).transpose(1, 2).reshape(B, -1, C)
        delta = s.o(a)
        p = torch.sigmoid(sc.float()).gather(1, idx).unsqueeze(-1)                       # score weights (gradient to candidate head)
        upd = (s.gamma * p * delta.float()).to(flat.dtype)
        out = flat.clone().scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, C), upd)
        return out.transpose(1, 2).reshape(B, C, Z, Y, X), cl


class FourierMix(nn.Module):
    """AFNO-style global token mixer: FFT over (z,y,x) -> block-diagonal complex 2-layer MLP -> soft-threshold -> iFFT, scaled residual."""
    def __init__(s, c, blocks=8, gamma0=0.3, thr=0.01):
        super().__init__()
        s.nb, s.bs, s.thr = blocks, c // blocks, thr
        s.norm = nn.InstanceNorm3d(c, affine=True)
        s.w1 = nn.Parameter(0.02 * torch.randn(2, blocks, s.bs, s.bs)); s.b1 = nn.Parameter(torch.zeros(2, blocks, s.bs))
        s.w2 = nn.Parameter(0.02 * torch.randn(2, blocks, s.bs, s.bs)); s.b2 = nn.Parameter(torch.zeros(2, blocks, s.bs))
        s.gamma = nn.Parameter(torch.tensor(float(gamma0)))
    def forward(s, x):
        B, C, Z, Y, X = x.shape
        with torch.autocast('cuda', enabled=False):
            f = torch.fft.rfftn(s.norm(x.float()), dim=(2, 3, 4), norm='ortho').reshape(B, s.nb, s.bs, Z, Y, -1)
            fr, fi = f.real, f.imag
            e = lambda t, w: torch.einsum('bnizyx,nio->bnozyx', t, w)
            bb = lambda b: b.view(1, s.nb, s.bs, 1, 1, 1)
            o1r = F.relu(e(fr, s.w1[0]) - e(fi, s.w1[1]) + bb(s.b1[0])); o1i = F.relu(e(fr, s.w1[1]) + e(fi, s.w1[0]) + bb(s.b1[1]))
            o2r = e(o1r, s.w2[0]) - e(o1i, s.w2[1]) + bb(s.b2[0]); o2i = e(o1r, s.w2[1]) + e(o1i, s.w2[0]) + bb(s.b2[1])
            o = torch.complex(F.softshrink(o2r, s.thr), F.softshrink(o2i, s.thr)).reshape(B, C, Z, Y, -1)
            y = torch.fft.irfftn(o, s=(Z, Y, X), dim=(2, 3, 4), norm='ortho')
        return x + s.gamma * y.to(x.dtype)


class LKMix(nn.Module):
    """control for FourierMix: large-kernel depthwise conv (3,7,7) + low-rank channel MLP, same scaled residual."""
    def __init__(s, c, r=64, gamma0=0.3):
        super().__init__()
        s.norm = nn.InstanceNorm3d(c, affine=True); s.dw = nn.Conv3d(c, c, (3, 7, 7), padding=(1, 3, 3), groups=c)
        s.pw1, s.pw2 = nn.Conv3d(c, r, 1), nn.Conv3d(r, c, 1); s.gamma = nn.Parameter(torch.tensor(float(gamma0)))
    def forward(s, x): return x + s.gamma * s.pw2(F.gelu(s.pw1(s.dw(s.norm(x)))))


class Net(nn.Module):
    def __init__(s, arm, k_frac=0.10, ch=(48, 96, 192, 320), n_cls=3):
        super().__init__()
        s.arm = arm
        m = re.match(r'^([A-Z]\d)(m)?(p)?(?:k(\d+))?(?:w(\d+))?$', arm)   # 'A2k3' = A2 with K=3% of cells; 'A6w5' = candidate-loss weight 0.5; 'A6m' = extra quarter-res candidate head
        s.aux_w, multi, s.use_gt = 0.2, False, True       # 'R2p' = routed decoder trained with predicted masks only (no ground-truth cells)
        if m:
            arm, multi = m.group(1), bool(m.group(2))
            s.use_gt = not m.group(3)
            if m.group(4): k_frac = int(m.group(4)) / 100
            if m.group(5): s.aux_w = int(m.group(5)) / 10
        s.base = arm
        s.stem = nn.Conv3d(1, ch[0], (1, 3, 3), padding=(0, 1, 1))
        s.e0 = MSBlock(ch[0], ch[0], 1)
        s.d1 = nn.Conv3d(ch[0], ch[1], (1, 2, 2), stride=(1, 2, 2)); s.e1 = MSBlock(ch[1], ch[1], 1)
        s.d2 = nn.Conv3d(ch[1], ch[2], 2, stride=2); s.e2 = MSBlock(ch[2], ch[2], 3)
        s.d3 = nn.Conv3d(ch[2], ch[3], 2, stride=2); s.b = MSBlock(ch[3], ch[3], 3)
        s.u2 = MSBlock(ch[3] + ch[2], ch[2], 3); s.u1 = MSBlock(ch[2] + ch[1], ch[1], 1); s.u0 = MSBlock(ch[1] + ch[0], ch[0], 1)
        mode = {'A0': None, 'A2': 'topk', 'A4': 'random', 'A5': 'dense', 'A6': 'none', 'A7': 'none', 'D0': None, 'C1': None, 'C2': None, 'R2': None, 'B2': 'topk', 'B4': 'random', 'B5': 'dense', 'F1': 'none', 'F0': 'none'}[arm]
        s.ctx_from_bottleneck = arm.startswith('B')
        s.ref = (SparseRefine(ch[1], mode, k_frac, c_ctx=ch[3], gamma0=1.0) if s.ctx_from_bottleneck else SparseRefine(ch[1], mode, k_frac)) if mode else None
        s.mix = FourierMix(ch[3]) if arm == 'F1' else (LKMix(ch[3]) if arm == 'F0' else None)
        s.cand2 = nn.Conv3d(ch[2], 1, 1) if multi else None; s.cl2 = None
        s.head = nn.Conv3d(ch[0], n_cls, 1)
        s.ds1 = nn.Conv3d(ch[1], n_cls, 1) if arm == 'D0' else None
        s.ds2 = None
        s.ds_out = None
        s.early = None
        if arm == 'C1': s.early = nn.Conv3d(ch[1], 1, 1)
        s.gt_cells, s.act_sum, s.act_n, s.tau = None, 0.0, 0, 0.25
        if arm in ('C2', 'R2'): s.early = nn.Sequential(MSBlock(ch[1] + ch[2], ch[1], 1), MSBlock(ch[1], ch[1], 1), nn.Conv3d(ch[1], 1, 1))
    def forward_routed(s, x0, x1, x2, y2, up):
        cl = s.early(torch.cat([x1, up(x2, x1)], 1))                                  # candidate logits on the cell grid
        act = torch.sigmoid(cl.detach().float()) > s.tau
        if s.training and s.use_gt and s.gt_cells is not None: act = act | (s.gt_cells > 0)       # train: also include ground-truth vessel cells
        m1 = F.max_pool3d(act.float(), 3, 1, 1)                                       # dilate by one cell
        m0 = m1.repeat_interleave(2, 3).repeat_interleave(2, 4)[..., :x0.shape[3], :x0.shape[4]]
        if not s.training: s.act_sum += float(m1.sum()); s.act_n += m1.numel()
        y1 = s.u1(torch.cat([up(y2, x1), x1], 1) * m1, m1)
        y0 = s.u0(torch.cat([up(y1, x0), x0], 1) * m0, m0)
        return s.head(y0), cl

    def forward(s, x):
        x0 = s.e0(F.relu(s.stem(x))); x1 = s.e1(s.d1(x0)); x2 = s.e2(s.d2(x1)); xb = s.b(s.d3(x2))
        if s.mix is not None: xb = s.mix(xb)
        up = lambda t, ref: F.interpolate(t, size=ref.shape[2:], mode='trilinear', align_corners=False)
        y2 = s.u2(torch.cat([up(xb, x2), x2], 1))
        if s.base == 'R2': return s.forward_routed(x0, x1, x2, y2, up)
        y1 = s.u1(torch.cat([up(y2, x1), x1], 1))
        cl = None
        if s.cand2 is not None: s.cl2 = s.cand2(y2)
        if s.early is not None: cl = s.early(x1) if s.arm == 'C1' else s.early(torch.cat([x1, up(x2, x1)], 1))
        if s.ds1 is not None and s.training: s.ds_out = (s.ds1(y1),)
        if s.ref is not None: y1, cl = s.ref(y1, xb if s.ctx_from_bottleneck else None)
        y0 = s.u0(torch.cat([up(y1, x0), x0], 1))
        return s.head(y0), cl


# ----------------------------------------------------------------------------- losses
def seg_loss(logits, tgt):
    ce = F.cross_entropy(logits.float(), tgt)
    p = F.softmax(logits.float(), 1); oh = F.one_hot(tgt, 3).permute(0, 4, 1, 2, 3).float(); terms = []
    for c in (1, 2):
        inter = (p[:, c] * oh[:, c]).flatten(1).sum(1); den = p[:, c].flatten(1).sum(1) + oh[:, c].flatten(1).sum(1)
        valid = oh[:, c].flatten(1).sum(1) > 0
        terms.append(torch.where(valid, 1 - (2 * inter + 1e-5) / (den + 1e-5), torch.zeros_like(inter)))
    t = torch.stack(terms, 1); n = (torch.stack([(oh[:, c].flatten(1).sum(1) > 0) for c in (1, 2)], 1)).float().sum().clamp(min=1)
    return 0.7 * t.sum() / n + 0.3 * ce, ce.item()


def cand_target(tgt, shape):
    u = (tgt > 0).float().unsqueeze(1); return F.adaptive_max_pool3d(u, shape)


def pool_label(y, k):
    oh = F.one_hot(y, 3).permute(0, 4, 1, 2, 3)[:, 1:].float(); pooled = F.max_pool3d(oh, k, k); has = pooled.sum(1) > 0
    return torch.where(has, pooled.argmax(1) + 1, torch.zeros(has.shape, dtype=torch.long, device=y.device)).long()


def cand_loss(cl, tgt, tversky=True):
    t = cand_target(tgt, cl.shape[2:]); c = cl.float(); pos = t.sum(); neg = t.numel() - pos
    bce = F.binary_cross_entropy_with_logits(c, t, pos_weight=(neg / (pos + 1)).clamp(1, 100))
    p = torch.sigmoid(c); tp = (p * t).sum(); fp = (p * (1 - t)).sum(); fn = ((1 - p) * t).sum()
    tv = 1 - (tp + 1e-5) / (tp + 0.3 * fp + 0.7 * fn + 1e-5)
    return bce + tv if tversky else bce


# ----------------------------------------------------------------------------- metrics
def surface(m): return m & ~ndi.binary_erosion(m, iterations=1)

def crop_union(a, b, margin=3):
    idx = np.argwhere(a | b)
    if len(idx) == 0: return None
    lo = np.maximum(idx.min(0) - margin, 0); hi = np.minimum(idx.max(0) + 1 + margin, a.shape)
    return tuple(slice(int(x), int(y)) for x, y in zip(lo, hi))

def metrics_binary(pred, gt, spacing):
    out = dict(dice=float('nan'), cldice=float('nan'), asd=float('nan'), hd95=float('nan'))
    sl = crop_union(pred, gt)
    if sl is None: return out
    p, g = pred[sl], gt[sl]
    out['dice'] = float(2 * (p & g).sum() / max(1, p.sum() + g.sum()))
    if p.sum() == 0 or g.sum() == 0: return out
    sp_, sg_ = skeletonize(p), skeletonize(g)
    tprec = (sp_ & g).sum() / max(1, sp_.sum()); tsens = (sg_ & p).sum() / max(1, sg_.sum())
    out['cldice'] = float(2 * tprec * tsens / max(1e-8, tprec + tsens))
    ps, gs = surface(p), surface(g)
    d_to_g = ndi.distance_transform_edt(~gs, sampling=spacing); d_to_p = ndi.distance_transform_edt(~ps, sampling=spacing)
    a, b = d_to_g[ps], d_to_p[gs]
    out['asd'] = float((a.mean() + b.mean()) / 2); out['hd95'] = float(max(np.percentile(a, 95), np.percentile(b, 95)))
    return out


@torch.no_grad()
def predict_volume(net, vol, stride=(16, 48, 48)):
    net.eval(); Z, Y, X = vol.shape; pad = [max(0, w - s) for w, s in zip(PATCH, vol.shape)]
    v = np.pad(vol, [(0, p) for p in pad], mode='constant', constant_values=float(vol.min()))
    Zp, Yp, Xp = v.shape
    def starts(L, w, st): s = list(range(0, max(1, L - w + 1), st)); return s if s[-1] == L - w or L <= w else s + [L - w]
    sig = [w / 8 for w in PATCH]; grid = np.meshgrid(*[np.arange(w) - (w - 1) / 2 for w in PATCH], indexing='ij')
    gw = np.exp(-sum((g / s) ** 2 for g, s in zip(grid, sig)) / 2); gw = torch.from_numpy(np.maximum(gw / gw.max(), 1e-3)).float().to(dev)
    acc = torch.zeros(3, Zp, Yp, Xp, device=dev); cnt = torch.zeros(Zp, Yp, Xp, device=dev)
    vt = torch.from_numpy(v.astype(np.float32)).to(dev)
    wins = [(a, b, c) for a in starts(Zp, PATCH[0], stride[0]) for b in starts(Yp, PATCH[1], stride[1]) for c in starts(Xp, PATCH[2], stride[2])]
    for i in range(0, len(wins), 4):
        chunk = wins[i:i + 4]; x = torch.stack([vt[a:a + PATCH[0], b:b + PATCH[1], c:c + PATCH[2]] for a, b, c in chunk]).unsqueeze(1)
        with torch.autocast('cuda', dtype=torch.float16): lg, _ = net(x)
        pr = F.softmax(lg.float(), 1)
        for (a, b, c), p in zip(chunk, pr):
            acc[:, a:a + PATCH[0], b:b + PATCH[1], c:c + PATCH[2]] += p * gw; cnt[a:a + PATCH[0], b:b + PATCH[1], c:c + PATCH[2]] += gw
    assert float(cnt.min()) > 0, 'uncovered voxels'
    return (acc / cnt).argmax(0)[:Z, :Y, :X].cpu().numpy().astype(np.uint8), len(wins)


@torch.no_grad()
def candidate_coverage(net, data, cid, n=48, seed=123):
    """per-patch recall of union-vessel and centreline positions inside the top-k% candidate set (k = 1/5/10%) at the refinement resolution."""
    if net.ref is None and net.early is None: return None
    net.eval(); rng = np.random.default_rng(seed); res = {b: [0, 0, 0, 0] for b in (0.01, 0.05, 0.10)}
    for j in range(n):
        c = (1 if j % 3 == 0 else 2 if j % 3 == 1 else 0)
        pts = data.coords[cid].get(c) if c else None
        center = pts[rng.integers(len(pts))] if (pts is not None and len(pts)) else [rng.integers(s) for s in data.vol[cid].shape]
        im, lb = data.extract(cid, center); x = torch.from_numpy(im).float().to(dev)[None, None]
        with torch.autocast('cuda', dtype=torch.float16): _, cl = net(x)
        t = cand_target(torch.from_numpy(lb).long().to(dev)[None], cl.shape[2:])[0, 0].flatten().bool()
        sk = torch.from_numpy(skeletonize(lb > 0).astype(np.uint8)).float().to(dev)[None, None]
        ts = F.adaptive_max_pool3d(sk, cl.shape[2:])[0, 0].flatten().bool()
        sc = cl.float().flatten(); N = sc.numel()
        for b in res:
            sel = torch.zeros(N, dtype=torch.bool, device=dev); sel[sc.topk(max(1, int(round(b * N)))).indices] = True
            res[b][0] += int((sel & t).sum()); res[b][1] += int(t.sum()); res[b][2] += int((sel & ts).sum()); res[b][3] += int(ts.sum())
    return {f'{int(b*100)}%': dict(vessel_cov=r[0] / max(1, r[1]), centerline_cov=r[2] / max(1, r[3])) for b, r in res.items()}


# ----------------------------------------------------------------------------- run
def run(arm, seed, epochs, ppc, tag, max_steps=None, bs=2, cases='cases12.json'):
    cs = json.load(open(cases)); tr, va = Data(cs['train']), Data(cs['val'])
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Net(arm).to(dev); npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4); scaler = torch.amp.GradScaler('cuda')
    steps_per_epoch = len(cs['train']) * ppc // bs; total = epochs * steps_per_epoch; warm = 2 * steps_per_epoch
    out = f'{RUNS}/{tag}'; os.makedirs(out, exist_ok=True); log = open(f'{out}/train_log.txt', 'a', encoding='utf-8')
    def P(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
    P(f'arm={arm} seed={seed} params={npar/1e6:.3f}M epochs={epochs} steps/epoch={steps_per_epoch} total={total}')
    step, t0 = 0, time.time(); torch.cuda.reset_peak_memory_stats()
    for ep in range(epochs):
        net.train(); order = [c for c in cs['train'] for _ in range(ppc)]; rng.shuffle(order); el = []
        for i in range(0, len(order) - bs + 1, bs):
            lr = 1e-3 * (step + 1) / warm if step < warm else 1e-6 + 0.5 * (1e-3 - 1e-6) * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g in opt.param_groups: g['lr'] = lr
            xs, ys = zip(*[tr.sample(rng, c) for c in order[i:i + bs]])
            x = torch.from_numpy(np.stack(xs)).float().to(dev).unsqueeze(1); y = torch.from_numpy(np.stack(ys)).long().to(dev)
            x = x * (1 + 0.1 * (torch.rand(bs, 1, 1, 1, 1, device=dev) * 2 - 1)) + 0.1 * (torch.rand(bs, 1, 1, 1, 1, device=dev) * 2 - 1)
            with torch.autocast('cuda', dtype=torch.float16):
                net.gt_cells = F.max_pool3d((y > 0).float().unsqueeze(1), (1, 2, 2))
                lg, cl = net(x)
            loss, ce = seg_loss(lg, y); aux = cand_loss(cl, y, tversky=(arm != 'A7')) if cl is not None else torch.zeros((), device=dev)
            if net.cl2 is not None: aux = aux + cand_loss(net.cl2, y, tversky=True)
            tot = loss + net.aux_w * aux
            if net.ds_out is not None:      # matched generic supervision: soft class-fraction targets at the candidate-head level, weight 0.2
                frac = F.avg_pool3d(F.one_hot(y, 3).permute(0, 4, 1, 2, 3).float(), (1, 2, 2))
                tot = tot + 0.2 * (-(frac * F.log_softmax(net.ds_out[0].float(), 1)).sum(1).mean())
            opt.zero_grad(set_to_none=True); scaler.scale(tot).backward(); scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update()
            if not torch.isfinite(tot): P('NaN/inf loss, stopping'); return None
            el.append((float(loss), float(aux))); step += 1
            if max_steps and step >= max_steps: break
        P(f'ep {ep+1}/{epochs} seg_loss {np.mean([e[0] for e in el]):.4f} cand_loss {np.mean([e[1] for e in el]):.4f} lr {lr:.2e} t={time.time()-t0:.0f}s mem={torch.cuda.max_memory_allocated()/2**30:.2f}GB')
        if max_steps and step >= max_steps: break
    train_time = time.time() - t0
    if net.ref is not None:
        P('grad check: cand-head grad norm', float(net.ref.cand.weight.grad.norm()) if net.ref.cand.weight.grad is not None else None, '| gamma', float(net.ref.gamma))
    torch.save(net.state_dict(), f'{out}/model.pt')
    res = dict(arm=arm, seed=seed, params_M=npar / 1e6, train_s=train_time, peak_mem_GB=torch.cuda.max_memory_allocated() / 2**30, steps=step, cases={})
    if max_steps: P('smoke run finished', json.dumps(res)); return res
    # evaluation on validation cases (whole ROI volume, sliding window with full coverage)
    t1 = time.time()
    for cid in cs['val']:
        t2 = time.time(); net.act_sum, net.act_n = 0.0, 0; pred, nw = predict_volume(net, va.vol[cid]); gt = va.lab[cid]; act = net.act_sum / max(1, net.act_n)
        m = {'HV': metrics_binary(pred == 1, gt == 1, SPACING), 'PV': metrics_binary(pred == 2, gt == 2, SPACING), 'union': metrics_binary(pred > 0, gt > 0, SPACING)}
        m['coverage'] = candidate_coverage(net, va, cid); m['windows'] = nw; m['active_frac'] = act; m['eval_s'] = time.time() - t2; res['cases'][cid] = m
        P(cid, 'Dice HV %.3f PV %.3f union %.3f clDice %.3f ASD %.2f mm (%d windows, %.0fs)' % (m['HV']['dice'], m['PV']['dice'], m['union']['dice'], m['union']['cldice'], m['union']['asd'], nw, m['eval_s']))
    res['eval_s'] = time.time() - t1
    json.dump(res, open(f'{out}/results.json', 'w'), indent=1); P('done', f'train {train_time:.0f}s eval {res["eval_s"]:.0f}s'); return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='A2'); ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--epochs', type=int, default=24); ap.add_argument('--ppc', type=int, default=16); ap.add_argument('--tag', default=None)
    ap.add_argument('--max_steps', type=int, default=None); ap.add_argument('--cases', default='cases12.json'); a = ap.parse_args()
    run(a.arm, a.seed, a.epochs, a.ppc, a.tag or f'{a.arm}_s{a.seed}', a.max_steps, cases=a.cases)
