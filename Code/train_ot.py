# coding: utf-8
"""Optional unsupervised fine-tuning of IFCNN-OT with a Wasserstein objective.

Only CONV3/CONV4 and the global interpolation weight λ are trained; CONV1 (ResNet)
and CONV2 stay frozen so that the feature space in which OT is solved is fixed.

Loss on the fused image F given visible V and infrared I:
    L = L_pix + α L_grad + β L_ot
    target = w(V,I) I + (1-w(V,I)) V                         (activity target)
    L_pix  = || F - target ||_1                              (intensity fidelity)
    L_grad = || ∇F - ∇target ||_1                            (signed edge preservation)
    L_ot   = S_ε(φ(F), φ(V)) + S_ε(φ(F), φ(I))               (Sinkhorn divergence in feature space)

Example:
    cd Code
    python train_ot.py --epochs 50 --save snapshots/IFCNN-OT.pth
    python IFCNN_OT_Main.py --weights snapshots/IFCNN-OT.pth
"""
import os
import random
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model_ot import myIFCNN_OT
from ot_fusion import sinkhorn_divergence
from IFCNN_OT_Main import IV_FILENAMES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='datasets/IVDataset/')
    p.add_argument('--init', default='snapshots/IFCNN-MAX.pth')
    p.add_argument('--save', default='snapshots/IFCNN-OT.pth')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--crop', type=int, default=128)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--alpha', type=float, default=1.0, help='gradient-loss weight')
    p.add_argument('--beta', type=float, default=0.1, help='OT-loss weight')
    p.add_argument('--ot_points', type=int, default=256, help='features sampled for the OT loss')
    p.add_argument('--fuse_mode', default='adaptive', choices=['adaptive', 'interp', 'max'])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--train_names',
                   default='Camp1,Dune,Gun,Navi,Road2,Steamboat,T2,T3,Trees4906,Trees4917',
                   help='comma-separated training scenes; keep evaluation scenes disjoint')
    return p.parse_args()


def load_pairs(root, names):
    to_t = transforms.ToTensor()
    pairs = []
    unknown = sorted(set(names) - set(IV_FILENAMES))
    if unknown:
        raise ValueError('unknown training scenes: %s' % unknown)
    for n in names:
        vis = to_t(Image.open(os.path.join(root, '%s_Vis.png' % n)).convert('RGB'))
        ir = to_t(Image.open(os.path.join(root, '%s_IR.png' % n)).convert('RGB'))
        pairs.append((vis, ir))
    return pairs


def random_crops(pairs, crop, batch):
    vs, irs = [], []
    for _ in range(batch):
        vis, ir = random.choice(pairs)
        _, h, w = vis.shape
        y = random.randint(0, h - crop)
        x = random.randint(0, w - crop)
        vs.append(vis[:, y:y + crop, x:x + crop])
        irs.append(ir[:, y:y + crop, x:x + crop])
    return torch.stack(vs), torch.stack(irs)


def gradient(x):
    gx = x[:, :, :, 1:] - x[:, :, :, :-1]
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return gx, gy


def activity_target(vis, ir):
    """Construct an explicit, differentiable-free target from local source activity."""
    def activity(x):
        gray = x.mean(dim=1, keepdim=True)
        local = F.avg_pool2d(F.pad(gray, (2, 2, 2, 2), mode='replicate'), 5, stride=1)
        gx, gy = gradient(gray)
        gx = F.pad(gx.abs(), (0, 1, 0, 0))
        gy = F.pad(gy.abs(), (0, 0, 0, 1))
        return gx + gy + 0.5 * (gray - local).abs()

    scores = torch.cat((activity(vis), activity(ir)), dim=1)
    ir_weight = F.softmax(scores / 0.1, dim=1)[:, 1:2]
    return (1 - ir_weight) * vis + ir_weight * ir


def sample_points(feat, n):
    B, C, H, W = feat.shape
    flat = feat.flatten(2).transpose(1, 2)                         # (B, HW, C)
    idx = torch.randperm(H * W, device=feat.device)[:n]
    return flat[:, idx]


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = myIFCNN_OT(pretrained_ifcnn=args.init, fuse_mode=args.fuse_mode,
                       learn_lambda=True).to(args.device)
    for p in model.conv2.parameters():
        p.requires_grad = False
    model.conv1.eval()
    model.conv2.eval()
    params = list(model.conv3.parameters()) + list(model.conv4.parameters())
    if args.fuse_mode != 'max':
        params.append(model.lam_logit)
    opt = torch.optim.Adam(params, lr=args.lr)

    train_names = [n.strip() for n in args.train_names.split(',') if n.strip()]
    if not train_names:
        raise ValueError('--train_names must contain at least one scene')
    pairs = load_pairs(args.root, train_names)
    steps_per_epoch = max(1, len(pairs) // args.batch)

    for epoch in range(args.epochs):
        tot = 0.0
        for _ in range(steps_per_epoch):
            vis, ir = random_crops(pairs, args.crop, args.batch)
            vis, ir = vis.to(args.device), ir.to(args.device)

            out, aux = model(vis, ir, return_aux=True)

            target = activity_target(vis, ir)
            l_pix = F.l1_loss(out, target)
            gxo, gyo = gradient(out)
            gxt, gyt = gradient(target)
            l_grad = F.l1_loss(gxo, gxt) + F.l1_loss(gyo, gyt)

            f_out = model.extract(out)
            l_ot = (sinkhorn_divergence(sample_points(f_out, args.ot_points),
                                        sample_points(aux['f_vis'], args.ot_points).detach()) +
                    sinkhorn_divergence(sample_points(f_out, args.ot_points),
                                        sample_points(aux['f_ir'], args.ot_points).detach())).mean()

            loss = l_pix + args.alpha * l_grad + args.beta * l_ot
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        print('epoch %3d  loss %.4f  lambda %.3f' % (epoch + 1, tot / steps_per_epoch, model.lam.item()))

    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': opt.state_dict(),
        'model_config': model.config,
        'training_config': vars(args),
        'epoch': args.epochs,
        'seed': args.seed,
        'train_names': train_names,
    }, args.save)
    print('saved', args.save)


if __name__ == '__main__':
    main()
