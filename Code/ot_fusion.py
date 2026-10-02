'''---------------------------------------------------------------------------
Optimal-transport building blocks used by IFCNN-OT.

The core idea: treat the per-pixel feature vectors of the visible image and of
the infrared image as two discrete probability measures living in the
IFCNN feature space.  An entropic-regularised optimal transport plan between
them tells us, for every visible-image location, *which* infrared features
should be moved there.  The barycentric projection of that plan therefore
"transfers" the infrared features into the geometry of the visible image.
----------------------------------------------------------------------------'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_log(cost, row_marginal=None, col_marginal=None, eps=0.05, n_iters=50):
    """Entropic OT in the log domain (numerically stable Sinkhorn).

    Args:
        cost: (B, N, M) ground-cost matrices.
        row_marginal: (B, N) row marginals (defaults to uniform).
        col_marginal: (B, M) column marginals (defaults to uniform).
        eps:  entropic regularisation strength (relative to the cost scale).
        n_iters: number of Sinkhorn iterations.

    Returns:
        plan: (B, N, M) transport plan with row sums ~a and column sums ~b.
    """
    if eps <= 0:
        raise ValueError('eps must be positive')
    if n_iters < 1:
        raise ValueError('n_iters must be at least 1')
    B, N, M = cost.shape
    if row_marginal is None:
        log_a = torch.full((B, N), -math.log(N), device=cost.device, dtype=cost.dtype)
    else:
        log_a = torch.log(row_marginal.clamp_min(1e-12))
    if col_marginal is None:
        log_b = torch.full((B, M), -math.log(M), device=cost.device, dtype=cost.dtype)
    else:
        log_b = torch.log(col_marginal.clamp_min(1e-12))

    log_K = -cost / eps
    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)
    for _ in range(n_iters):
        f = log_a - torch.logsumexp(log_K + g.unsqueeze(1), dim=2)
        g = log_b - torch.logsumexp(log_K + f.unsqueeze(2), dim=1)
    return torch.exp(log_K + f.unsqueeze(2) + g.unsqueeze(1))


def sinkhorn_divergence(x, y, eps=0.05, n_iters=50):
    """Debiased Sinkhorn divergence S_eps(x, y) between two point clouds.

    x: (B, N, D), y: (B, M, D). Returns a (B,) tensor. Used as an
    unsupervised training objective: the fused features should be close, in
    the Wasserstein sense, to the feature distributions of both sources.
    """
    # Unit-normalising points gives all three terms the same bounded ground
    # metric. Independently normalising each cost matrix would not define one
    # Sinkhorn divergence.
    x = F.normalize(x, dim=-1, eps=1e-8)
    y = F.normalize(y, dim=-1, eps=1e-8)

    def ot_cost(p, q):
        B, N, M = p.shape[0], p.shape[1], q.shape[1]
        c = torch.cdist(p, q) ** 2
        row = torch.full((B, N), 1.0 / N, device=p.device, dtype=p.dtype)
        col = torch.full((B, M), 1.0 / M, device=p.device, dtype=p.dtype)
        plan = sinkhorn_log(c, row, col, eps=eps, n_iters=n_iters)
        reference = row.unsqueeze(2) * col.unsqueeze(1)
        # Regularised OT with KL(P || row x col), not only <P, C>.
        kl = plan * (torch.log(plan.clamp_min(1e-12)) -
                     torch.log(reference.clamp_min(1e-12)))
        return (plan * c + eps * kl).sum(dim=(1, 2))

    return ot_cost(x, y) - 0.5 * ot_cost(x, x) - 0.5 * ot_cost(y, y)


def window_partition(x, ws, stride):
    """(B, C, H, W) -> overlapping windows (B * L, ws*ws, C) and the padded size.

    Windows are extracted with `stride` (<= ws), so neighbouring windows overlap;
    the image is reflect-padded by ws - stride on every side so that each pixel
    is covered by the same number of windows.
    """
    B, C, H, W = x.shape
    pad = ws - stride
    pad_h = (stride - (H + 2 * pad - ws) % stride) % stride
    pad_w = (stride - (W + 2 * pad - ws) % stride) % stride
    x = F.pad(x, (pad, pad + pad_w, pad, pad + pad_h), mode='replicate')
    Hp, Wp = x.shape[-2:]
    cols = F.unfold(x, kernel_size=ws, stride=stride)               # (B, C*ws*ws, L)
    L = cols.shape[-1]
    cols = cols.view(B, C, ws * ws, L).permute(0, 3, 2, 1).reshape(B * L, ws * ws, C)
    return cols, (Hp, Wp)


def window_reverse(tokens, ws, stride, B, C, padded_hw, orig_hw, weights):
    """Inverse of window_partition with a per-window blending kernel."""
    Hp, Wp = padded_hw
    H, W = orig_hw
    pad = ws - stride
    L = tokens.shape[0] // B
    tokens = tokens * weights.view(1, ws * ws, 1)
    cols = tokens.view(B, L, ws * ws, C).permute(0, 3, 2, 1).reshape(B, C * ws * ws, L)
    out = F.fold(cols, output_size=(Hp, Wp), kernel_size=ws, stride=stride)
    ones = weights.view(1, ws * ws, 1).expand(B, ws * ws, L).reshape(B, ws * ws, L)
    norm = F.fold(ones, output_size=(Hp, Wp), kernel_size=ws, stride=stride)
    out = out / norm.clamp_min(1e-8)
    return out[:, :, pad:pad + H, pad:pad + W]


class FeatureTransport(nn.Module):
    """Transfers infrared features into the visible image via local OT.

    The image is covered by overlapping ws x ws windows.  Inside every window we
    solve an entropic OT problem between the visible feature vectors (target
    measure) and the infrared feature vectors (source measure).  The ground
    cost mixes a feature-space distance with a spatial distance so that the
    transported infrared content stays spatially coherent with the registered
    visible image.  The barycentric projection of the plan yields, for each
    visible pixel, an infrared feature "transported" to that location.
    Overlapping windows are blended with a Hann kernel to avoid block seams.
    """

    def __init__(self, window_size=16, eps=0.02, n_iters=50, spatial_weight=4.0,
                 saliency_marginals=False, overlap=2, detail_kernel=9,
                 hard_assignment=False, window_chunk_size=128):
        super(FeatureTransport, self).__init__()
        if window_size < 1:
            raise ValueError('window_size must be positive')
        if overlap < 1 or overlap > window_size or window_size % overlap:
            raise ValueError('overlap must divide window_size and be in [1, window_size]')
        if eps <= 0 or n_iters < 1:
            raise ValueError('eps must be positive and n_iters must be at least 1')
        if detail_kernel > 1 and detail_kernel % 2 == 0:
            raise ValueError('detail_kernel must be odd, 0, or 1')
        if window_chunk_size < 1:
            raise ValueError('window_chunk_size must be positive')
        self.ws = window_size
        self.stride = window_size // overlap
        self.eps = eps
        self.n_iters = n_iters
        self.spatial_weight = spatial_weight
        self.saliency_marginals = saliency_marginals
        self.hard_assignment = hard_assignment
        self.window_chunk_size = window_chunk_size
        # detail_kernel > 0: the OT output only provides the low-frequency
        # (smoothed) displacement of the infrared features, while the original
        # high-frequency infrared detail is kept.  Counteracts the averaging of
        # the entropic barycentric projection on registered image pairs.
        self.detail_kernel = detail_kernel
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing='ij'), dim=-1)
        coords = coords.reshape(-1, 2).float() / max(window_size - 1, 1)
        # (1, ws*ws, ws*ws) squared spatial distances, normalised to [0, 1]
        self.register_buffer('spatial_cost', (torch.cdist(coords, coords) ** 2) / 2.0,
                             persistent=False)
        hann = torch.hann_window(window_size + 2, periodic=False)[1:-1]
        self.register_buffer('blend', (hann[:, None] * hann[None, :]).reshape(-1),
                             persistent=False)

    @staticmethod
    def _saliency(feat):
        """Per-pixel saliency from the feature-energy deviation inside a window."""
        energy = feat.norm(dim=-1)                                  # (Bw, N)
        energy = energy - energy.mean(dim=1, keepdim=True)
        return F.softmax(energy / (energy.std(dim=1, keepdim=True, unbiased=False) + 1e-6),
                         dim=1)

    def forward(self, f_vis, f_ir, return_plan=False):
        if f_vis.shape != f_ir.shape:
            raise ValueError('visible and infrared feature maps must have identical shapes; '
                             'got %s and %s' % (tuple(f_vis.shape), tuple(f_ir.shape)))
        B, C, H, W = f_vis.shape
        ws, stride = self.ws, self.stride
        tv, padded = window_partition(f_vis, ws, stride)            # (Bw, N, C)
        ti, _ = window_partition(f_ir, ws, stride)

        transported_chunks, plan_chunks = [], []
        for start in range(0, tv.shape[0], self.window_chunk_size):
            v = tv[start:start + self.window_chunk_size]
            i = ti[start:start + self.window_chunk_size]
            feat_cost = torch.cdist(v, i) ** 2
            feat_cost = feat_cost / (feat_cost.mean(dim=(1, 2), keepdim=True) + 1e-8)
            cost = feat_cost + self.spatial_weight * self.spatial_cost
            col = self._saliency(i) if self.saliency_marginals else None
            plan = sinkhorn_log(cost, col_marginal=col, eps=self.eps,
                                n_iters=self.n_iters)
            if self.hard_assignment:
                # Row-wise argmax heuristic; unlike a capacity-constrained
                # assignment, this does not preserve the column marginal.
                idx = plan.argmax(dim=2, keepdim=True).expand(-1, -1, i.shape[-1])
                transported_chunk = torch.gather(i, 1, idx)
            else:
                row_mass = plan.sum(dim=2, keepdim=True).clamp_min(1e-12)
                transported_chunk = torch.bmm(plan, i) / row_mass
            transported_chunks.append(transported_chunk)
            if return_plan:
                plan_chunks.append(plan)
        transported = torch.cat(transported_chunks, dim=0)

        transported = window_reverse(transported, ws, stride, B, C, padded, (H, W), self.blend)
        if self.detail_kernel > 1:
            k = self.detail_kernel
            displacement = transported - f_ir
            displacement = F.avg_pool2d(F.pad(displacement, (k // 2,) * 4, mode='replicate'),
                                        k, stride=1)
            transported = f_ir + displacement
        return transported, (torch.cat(plan_chunks, dim=0) if return_plan else None)


class ChannelOT(nn.Module):
    """Exact 1-D optimal transport (quantile matching) applied per channel.

    Pushes every infrared feature channel onto the marginal distribution of the
    corresponding visible channel, i.e. a monotone OT map in 1-D.  Cheap, global,
    and useful as a pre-alignment step before the local Sinkhorn transport.
    """

    def forward(self, f_src, f_tgt):
        B, C, H, W = f_src.shape
        src = f_src.reshape(B, C, -1)
        tgt = f_tgt.reshape(B, C, -1)
        src_sorted, src_idx = src.sort(dim=-1)
        tgt_sorted, _ = tgt.sort(dim=-1)
        if tgt_sorted.shape[-1] != src_sorted.shape[-1]:
            tgt_sorted = F.interpolate(tgt_sorted, size=src_sorted.shape[-1],
                                       mode='linear', align_corners=True)
        out = torch.empty_like(src)
        out.scatter_(-1, src_idx, tgt_sorted)
        return out.view(B, C, H, W)
