# IFCNN
Project page of  "[IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network](https://www.sciencedirect.com/science/article/pii/S1566253518305505),  Information Fusion, 54 (2020) 99-118". 



### Requirements
- pytorch=0.4.1
- python=3.x
- torchvision
- numpy
- opencv-python
- jupyter notebook (optional)
- anaconda (suggeted)

### Configuration
```bash
# Create your virtual environment using anaconda
conda create -n IFCNN python=3.5

# Activate your virtual environment
conda activate IFCNN

# Install the required libraries
conda install pytorch=0.4.1 cuda80 -c pytorch
conda install torchvision numpy jupyter notebook
pip install opencv-python
```


### Usage
```bash
# Clone our code
git clone https://github.com/uzeful/IFCNN.git
cd IFCNN/Code

# Remember to activate your virtual enviroment before running our code
conda activate IFCNN

# Replicate our image method on fusing multiple types of images
python IFCNN_Main.py

# Or run code part by part in notebook
jupyter notebook IFCNN_Notebook.ipynb
```



### IFCNN-OT: fusion through optimal transport (infrared → visible)
`Code/model_ot.py` extends IFCNN with an optimal-transport (OT) fusion stage for infrared/visible image fusion.
The frozen IFCNN feature extractor (CONV1 + CONV2) and reconstruction layers (CONV3 + CONV4) are kept, and
the pretrained `IFCNN-MAX.pth` weights are reused, so no training is required.

Pipeline:

```
Vis ──► CONV1 ──► CONV2 ──► f_vis ────────────────────────────┐
                                                               ├─► OT fusion ─► CONV3 ─► CONV4 ─► fused
IR  ──► CONV1 ──► CONV2 ──► f_ir ──► FeatureTransport ─► T(f_ir) ┘
```

1. **FeatureTransport** (`ot_fusion.FeatureTransport`): inside overlapping local windows, the per-pixel visible and
   infrared feature vectors are treated as two discrete measures and an entropic OT plan `P` between them is solved
   with log-domain Sinkhorn. The ground cost mixes feature-space distance and spatial distance (`--spatial_weight`),
   so the transport stays coherent with the registered visible image. The barycentric projection of the plan,
   `T(f_ir) = (P f_ir) / (P 1)`, *transfers the infrared features into the geometry of the visible image*. Because the
   entropic projection averages features, only its smoothed displacement is applied while the infrared
   high-frequency detail is kept (`--detail_kernel`); `--hard_assignment` uses the argmax (Monge-style) map instead.
   Windows are blended with a Hann kernel to avoid seams.
2. **Fusion rule** (`IFCNN_OT.fuse`): a Wasserstein displacement interpolation between the visible features and
   the transported infrared features, `f = (1-λ) f_vis + λ · max(f_vis, T(f_ir))`, where λ is either a scalar
   (`--fuse_mode interp`) or a spatially-adaptive weight that grows where the transported infrared response is
   stronger than the visible one (`--fuse_mode adaptive`, default). `--fuse_mode max` gives plain IFCNN-MAX over
   `(f_vis, T(f_ir))`.
3. **ChannelOT** (`ot_fusion.ChannelOT`, optional, `--channel_ot`): exact 1-D optimal transport (quantile matching)
   per feature channel, pushing the marginal distribution of every infrared channel onto that of the visible channel
   before the local transport.

Metrics on the IV pairs *Camp / Road / Kayak / Octec* (mean; EN entropy, SD std-dev, SF spatial frequency,
MI = MI(Vis,F)+MI(IR,F), Qabf edge-transfer):

| model | EN | SD | SF | MI | Qabf |
|---|---|---|---|---|---|
| IFCNN-MAX (original) | 6.42 | 28.97 | 10.16 | 2.80 | 0.544 |
| IFCNN-OT `--fuse_mode max` | 6.41 | 28.81 | 10.07 | 2.75 | 0.530 |
| IFCNN-OT `--fuse_mode adaptive` (default) | 6.36 | 34.32 | 9.41 | 3.26 | 0.468 |

The adaptive OT fusion keeps the visible image as the base and injects thermal content where it is informative,
which raises contrast (SD) and information preserved from the sources (MI); plain max-fusion of the transported
features matches IFCNN-MAX. Fused results for the 14 IV pairs are in [Results/IV-OT](Results/IV-OT).

Usage:
```bash
cd Code
python IFCNN_OT_Main.py                                  # fuse the 14 IV pairs -> results/IFCNN-OT-ADAPTIVE-IV-*.png
python IFCNN_OT_Main.py --fuse_mode interp --lam 0.4     # fixed interpolation weight
python IFCNN_OT_Main.py --vis a_vis.png --ir a_ir.png --out fused.png
python IFCNN_OT_Main.py --save_aux                       # also save the OT weight map and transported-IR energy
python train_ot.py --epochs 50                           # optional: fine-tune CONV3/4 + λ with a Sinkhorn-divergence loss
python IFCNN_OT_Main.py --weights snapshots/IFCNN-OT.pth
```
Main knobs: `--window_size` (OT window), `--eps` (entropic regularisation), `--n_iters` (Sinkhorn iterations),
`--spatial_weight` (how strongly transport is kept spatially local), `--overlap` (window overlap factor), `--lam`
and `--gain` (operating point / sharpness of the adaptive weight). The code runs on CUDA when available and falls
back to CPU.

`ot_fusion.sinkhorn_divergence` additionally provides a debiased Sinkhorn divergence that `train_ot.py` uses as an
unsupervised objective: the fused image's features should be close, in the Wasserstein sense, to the feature
distributions of both source images.

### Typos
1. Eq. (4) in our paper is wrongly written, the correct expression can be referred to the official expression in [OpenCV document](https://docs.opencv.org/3.4.2/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa), i.e., <img src="https://latex.codecogs.com/gif.latex?G(i)=\alpha&space;\cdot&space;e^{-\frac{[i-(ksize-1)/2]^2}{2\sigma^2}}" title="G(i)=\alpha \cdot e^{-\frac{[i-(ksize-1)/2]^2}{2\sigma^2}}" />, where <img src="https://latex.codecogs.com/gif.latex?i=0&space;\cdots&space;(ksize-1)" title="i=0 \cdots (ksize-1)" />, <img src="https://latex.codecogs.com/gif.latex?ksize=2\times{kr}&plus;1" title="ksize=2\times{kr}+1" />, <img src="https://latex.codecogs.com/gif.latex?\sigma=0.6\times(ksize-1)&plus;0.8" title="\sigma=0.6\times(ksize-1)+0.8" />, and <img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /> is the scale factor chosen for achieving <img src="https://latex.codecogs.com/gif.latex?\sum&space;G\left(i\right)=1" title="\sum G\left(i\right)=1" />.
2. Stride and padding parameters of CONV4 are respectively 1 and 0, rather than both 0.



### Highlights
- Propose a general image fusion framework based on convolutional neural network
- Demonstrate good generalization ability for fusing various types of images
- Perform comparably or even better than other algorithms on four image datasets
- Create a large-scale and diverse multi-focus image dataset for training CNN models
- Incorporate perceptual loss to boost the model’s performance



### Architecture of our image fusion model
![flowchart](https://github.com/uzeful/IFCNN/blob/master/flowchart.png)



### Comparison Examples
1. Multi-focus image fusion
![CMF05](https://github.com/uzeful/IFCNN/blob/master/Comparisons/CMF05.png)


2. Infrared and visual image fusion
![CMF05](https://github.com/uzeful/IFCNN/blob/master/Comparisons/IVroad.png)


3. Multi-modal medical image fusion
![MDc02](https://github.com/uzeful/IFCNN/blob/master/Comparisons/MDc02.png)


4. Multi-exposure image fusion
![MEdoor](https://github.com/uzeful/IFCNN/blob/master/Comparisons/MEdoor.png)



### Other Results of Our Model
1. Multi-focus image dataset: [Results/CMF](https://github.com/uzeful/IFCNN/tree/master/Results/CMF)
2. Infrared and visual image dataset: [Results/IV](https://github.com/uzeful/IFCNN/tree/master/Results/IV)
3. Multi-modal medical image dataset: [Results/MD](https://github.com/uzeful/IFCNN/tree/master/Results/MDDataset)
4. Multi-exposure image dataset: [Results/ME](https://github.com/uzeful/IFCNN/tree/master/Results/ME)



### Citation
If you find this code is useful for your research, please consider to cite our paper. Yu Zhang, Yu Liu, Peng Sun, Han Yan, Xiaolin Zhao, Li Zhang, [IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network](https://authors.elsevier.com/a/1ZTXt5a7-GbZZX),  Information Fusion, 54 (2020) 99-118.

```
@article{zhang2020IFCNN,
  title={IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network},
  author={Zhang, Yu and Liu, Yu and Sun, Peng and Yan, Han and Zhao, Xiaolin and Zhang, Li},
  journal={Information Fusion},
  volume={54},
  pages={99--118},
  year={2020},
  publisher={Elsevier}
}
```
