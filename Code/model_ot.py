'''---------------------------------------------------------------------------
IFCNN-OT: IFCNN with an optimal-transport fusion stage for infrared/visible
image fusion.

Pipeline (see model.py for the original IFCNN):

    Vis ──► CONV1 ──► CONV2 ──► f_vis ─────────────────┐
                                                        ├─► OT fusion ─► CONV3 ─► CONV4 ─► fused
    IR  ──► CONV1 ──► CONV2 ──► f_ir ─► [ChannelOT] ───┘

OT fusion: the infrared features are transported onto the visible image's
feature distribution (local entropic OT, barycentric projection), then the
fused feature is the Wasserstein displacement interpolation

    f_fused = (1 - λ) * f_vis + λ * T(f_ir)

where T(f_ir) are the transported infrared features and λ is either a fixed
scalar or a spatially-adaptive weight derived from infrared saliency.  The
feature extractor and reconstruction layers are shared with IFCNN, so the
pretrained IFCNN-MAX weights can be loaded directly.
----------------------------------------------------------------------------'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model import IFCNN
from ot_fusion import FeatureTransport, ChannelOT


class IFCNN_OT(IFCNN):
    """IFCNN whose feature-fusion rule is replaced by optimal transport."""

    FUSE_MODES = ('interp', 'adaptive', 'max')

    def __init__(self, resnet, fuse_mode='adaptive', lam=0.5, window_size=16,
                 eps=0.02, n_iters=50, spatial_weight=4.0, channel_ot=False,
                 saliency_marginals=False, overlap=2, detail_kernel=9, hard_assignment=False,
                 gain=6.0, smooth_kernel=5, learn_lambda=False):
        super(IFCNN_OT, self).__init__(resnet, fuse_scheme=0)
        assert fuse_mode in self.FUSE_MODES, 'fuse_mode must be one of %s' % (self.FUSE_MODES,)
        self.fuse_mode = fuse_mode
        self.use_channel_ot = channel_ot
        self.gain = gain
        self.smooth_kernel = smooth_kernel
        self.channel_ot = ChannelOT()
        self.transport = FeatureTransport(window_size=window_size, eps=eps, n_iters=n_iters,
                                          spatial_weight=spatial_weight,
                                          saliency_marginals=saliency_marginals, overlap=overlap,
                                          detail_kernel=detail_kernel,
                                          hard_assignment=hard_assignment)
        lam_t = torch.tensor(float(lam))
        if learn_lambda:
            self.lam_logit = nn.Parameter(torch.logit(lam_t.clamp(1e-3, 1 - 1e-3)))
        else:
            self.register_buffer('lam_logit', torch.logit(lam_t.clamp(1e-3, 1 - 1e-3)))

    @property
    def lam(self):
        return torch.sigmoid(self.lam_logit)

    def extract(self, x):
        out = F.pad(x, (3, 3, 3, 3), mode='replicate')
        out = self.conv1(out)
        out = self.conv2(out)
        return out

    def _adaptive_lambda(self, f_vis, f_ir_t, lam):
        """Spatially varying interpolation weight in [0, 1].

        Locations where the transported infrared response is stronger than the
        visible response (thermal targets) receive a larger weight; elsewhere the
        visible image dominates.  `lam` acts as the global operating point.
        """
        e_vis = f_vis.norm(dim=1, keepdim=True)
        e_ir = f_ir_t.norm(dim=1, keepdim=True)
        ratio = (e_ir - e_vis) / (e_ir + e_vis + 1e-6)             # in [-1, 1]
        k = self.smooth_kernel
        ratio = F.avg_pool2d(F.pad(ratio, (k // 2,) * 4, mode='replicate'), k, stride=1)
        return torch.sigmoid(torch.logit(lam) + self.gain * ratio)

    def fuse(self, f_vis, f_ir):
        if self.use_channel_ot:
            f_ir = self.channel_ot(f_ir, f_vis)
        f_ir_t, plan = self.transport(f_vis, f_ir)

        if self.fuse_mode == 'max':
            fused = torch.max(f_vis, f_ir_t)
            weight = None
        elif self.fuse_mode == 'interp':
            weight = self.lam
            fused = (1 - weight) * f_vis + weight * f_ir_t
        else:
            # adaptive: displacement interpolation between the visible features
            # and the element-wise max of (visible, transported infrared).  The
            # max keeps the contrast of whichever modality is locally stronger,
            # while the weight decides how much of the infrared content is
            # carried into the visible image at each location.
            weight = self._adaptive_lambda(f_vis, f_ir_t, self.lam)
            fused = (1 - weight) * f_vis + weight * torch.max(f_vis, f_ir_t)
        return fused, {'transported': f_ir_t, 'weight': weight, 'plan': plan}

    def forward(self, vis, ir, return_aux=False):
        f_vis = self.extract(vis)
        f_ir = self.extract(ir)
        fused, aux = self.fuse(f_vis, f_ir)
        out = self.conv3(fused)
        out = self.conv4(out)
        if return_aux:
            aux.update({'f_vis': f_vis, 'f_ir': f_ir, 'fused': fused})
            return out, aux
        return out


def myIFCNN_OT(pretrained_ifcnn='snapshots/IFCNN-MAX.pth', **kwargs):
    """Builds IFCNN-OT, initialising CONV1-4 from the pretrained IFCNN weights."""
    try:
        resnet = models.resnet101(weights=None)
    except TypeError:  # torchvision < 0.13
        resnet = models.resnet101(pretrained=False)
    model = IFCNN_OT(resnet, **kwargs)
    if pretrained_ifcnn is not None:
        state = torch.load(pretrained_ifcnn, map_location='cpu')
        missing, unexpected = model.load_state_dict(state, strict=False)
        unexpected = [k for k in unexpected]
        if unexpected:
            raise RuntimeError('Unexpected keys in IFCNN checkpoint: %s' % unexpected)
    return model
