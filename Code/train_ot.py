# coding: utf-8
"""Optional unsupervised fine-tuning of IFCNN-OT with a Wasserstein objective.

Only CONV3/CONV4 and the global interpolation weight λ are trained; CONV1 (ResNet)
and CONV2 stay frozen so that the feature space in which OT is solved is fixed.

Loss on the fused image F given visible V and infrared I:
    L = L_pix + α L_grad + β L_ot
    L_pix  = || F - V ||_1 + || F - max(V, I) ||_1            (intensity fidelity)
    L_grad = || ∇F - max(|∇V|, |∇I|) ||_1                    (edge preservation)
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
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--alpha', type=float, default=1.0, help='gradient-loss weight')
    p.add_argument('--beta', type=float, default=0.1, help='OT-loss weight')
    p.add_argument('--ot_points', type=int, default=1024, help='features sampled for the OT loss')
    p.add_argument('--fuse_mode', default='adaptive', choices=['adaptive', 'interp', 'max'])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def load_pairs(root):
    to_t = transforms.ToTensor()
    pairs = []
    for n in IV_FILENAMES:
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
    params = list(model.conv3.parameters()) + list(model.conv4.parameters()) + [model.lam_logit]
    opt = torch.optim.Adam(params, lr=args.lr)

    pairs = load_pairs(args.root)
    steps_per_epoch = max(1, len(pairs) // args.batch)

    for epoch in range(args.epochs):
        tot = 0.0
        for _ in range(steps_per_epoch):
            vis, ir = random_crops(pairs, args.crop, args.batch)
            vis, ir = vis.to(args.device), ir.to(args.device)

            out, aux = model(vis, ir, return_aux=True)

            l_pix = F.l1_loss(out, vis) + F.l1_loss(out, torch.max(vis, ir))
            gxo, gyo = gradient(out)
            gxv, gyv = gradient(vis)
            gxi, gyi = gradient(ir)
            l_grad = F.l1_loss(gxo.abs(), torch.max(gxv.abs(), gxi.abs())) + \
                F.l1_loss(gyo.abs(), torch.max(gyv.abs(), gyi.abs()))

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
    torch.save(model.state_dict(), args.save)
    print('saved', args.save)


if __name__ == '__main__':
    main()
