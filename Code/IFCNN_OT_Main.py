# coding: utf-8
"""Fuse infrared and visible images with IFCNN-OT.

Example:
    cd Code
    python IFCNN_OT_Main.py                       # all 14 IV pairs, adaptive OT fusion
    python IFCNN_OT_Main.py --fuse_mode interp --lam 0.4
    python IFCNN_OT_Main.py --vis my_vis.png --ir my_ir.png --out fused.png
    python IFCNN_OT_Main.py --save_aux            # also dump the OT weight / transported maps
"""
import os
import time
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model_ot import myIFCNN_OT
from utils.myDatasets import ImagePair
from utils.myTransforms import denorm

IV_FILENAMES = ['Camp', 'Camp1', 'Dune', 'Gun', 'Navi', 'Kayak', 'Octec', 'Road', 'Road2',
                'Steamboat', 'T2', 'T3', 'Trees4906', 'Trees4917']


def parse_args():
    p = argparse.ArgumentParser(description='IFCNN-OT infrared/visible fusion')
    p.add_argument('--root', default='datasets/IVDataset/')
    p.add_argument('--results', default='results/')
    p.add_argument('--weights', default='snapshots/IFCNN-MAX.pth',
                   help='IFCNN checkpoint used to initialise CONV1-4 (or an IFCNN-OT checkpoint)')
    p.add_argument('--vis', default=None, help='single visible image (overrides dataset loop)')
    p.add_argument('--ir', default=None, help='single infrared image')
    p.add_argument('--out', default=None, help='output path for single-pair mode')
    p.add_argument('--fuse_mode', default='adaptive', choices=['adaptive', 'interp', 'max'])
    p.add_argument('--lam', type=float, default=0.5, help='global IR interpolation weight')
    p.add_argument('--window_size', type=int, default=16)
    p.add_argument('--eps', type=float, default=0.02, help='entropic regularisation')
    p.add_argument('--n_iters', type=int, default=50, help='Sinkhorn iterations')
    p.add_argument('--spatial_weight', type=float, default=4.0,
                   help='weight of the spatial term in the ground cost (keeps transport local)')
    p.add_argument('--overlap', type=int, default=2, help='window overlap factor (1 = none)')
    p.add_argument('--detail_kernel', type=int, default=9,
                   help='smooth the OT displacement with this kernel and keep IR detail (0 = off)')
    p.add_argument('--hard_assignment', action='store_true',
                   help='use the argmax (Monge-style) map instead of the barycentric projection')
    p.add_argument('--gain', type=float, default=6.0, help='sharpness of the adaptive weight')
    p.add_argument('--channel_ot', action='store_true',
                   help='pre-align IR feature channels to the visible ones with 1-D OT')
    p.add_argument('--saliency_marginals', action='store_true',
                   help='weight the infrared marginal by thermal saliency')
    p.add_argument('--save_aux', action='store_true', help='save OT weight map and transported IR')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--no_save', action='store_true')
    return p.parse_args()


def build_model(args):
    model = myIFCNN_OT(pretrained_ifcnn=args.weights, fuse_mode=args.fuse_mode, lam=args.lam,
                       window_size=args.window_size, eps=args.eps, n_iters=args.n_iters,
                       spatial_weight=args.spatial_weight, channel_ot=args.channel_ot,
                       saliency_marginals=args.saliency_marginals, overlap=args.overlap,
                       detail_kernel=args.detail_kernel, hard_assignment=args.hard_assignment,
                       gain=args.gain)
    return model.eval().to(args.device)


def to_uint8(tensor, mean=(0, 0, 0), std=(1, 1, 1)):
    res = denorm(list(mean), list(std), tensor.clone()).clamp(0, 1) * 255
    return res.cpu().numpy().astype('uint8').transpose([1, 2, 0])


def fuse_pair(model, path_vis, path_ir, device, save_aux=False):
    loader = ImagePair(impath1=path_vis, impath2=path_ir,
                       transform=transforms.Compose([transforms.ToTensor()]))
    vis, ir = loader.get_pair()
    vis = vis.unsqueeze(0).to(device)
    ir = ir.unsqueeze(0).to(device)
    with torch.no_grad():
        out, aux = model(vis, ir, return_aux=True)
    fused = cv2.cvtColor(to_uint8(out[0]), cv2.COLOR_RGB2GRAY)
    extras = {}
    if save_aux:
        if aux['weight'] is not None and aux['weight'].dim() == 4:
            extras['weight'] = (aux['weight'][0, 0].clamp(0, 1) * 255).cpu().numpy().astype('uint8')
        t = aux['transported'][0].norm(dim=0)
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        extras['transported'] = (t * 255).cpu().numpy().astype('uint8')
    return fused, extras


def main():
    args = parse_args()
    model = build_model(args)
    tag = 'IFCNN-OT-%s' % args.fuse_mode.upper()
    os.makedirs(args.results, exist_ok=True)

    if args.vis and args.ir:
        pairs = [(os.path.splitext(os.path.basename(args.vis))[0], args.vis, args.ir)]
    else:
        pairs = [(n, os.path.join(args.root, '%s_Vis.png' % n), os.path.join(args.root, '%s_IR.png' % n))
                 for n in IV_FILENAMES]

    begin = time.time()
    for name, path_vis, path_ir in pairs:
        fused, extras = fuse_pair(model, path_vis, path_ir, args.device, save_aux=args.save_aux)
        if args.no_save:
            continue
        out_path = args.out if (args.out and len(pairs) == 1) else \
            os.path.join(args.results, '%s-IV-%s.png' % (tag, name))
        Image.fromarray(fused).save(out_path, format='PNG', compress_level=0)
        for k, v in extras.items():
            Image.fromarray(v).save(os.path.join(args.results, '%s-IV-%s-%s.png' % (tag, name, k)),
                                    format='PNG', compress_level=0)
        print('saved', out_path)
    print('Total processing time of IV dataset (%d pairs): %.3fs' % (len(pairs), time.time() - begin))


if __name__ == '__main__':
    main()
