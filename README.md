# IFCNN
Project page of  "IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network,  Information Fusion, 54 (2020) 99-118". 



# Requirements:
1. The code is tested with Ubuntu 14.04, cuda-8.0 and pytorch 0.4.1
2. The required packages include pytorch 0.4.1, torchvision, opencv-python, numpy, jupyter notebook


# Usage:
1. Directly run "python IFCNN_Main.py" to replicate our image fusion method
2. Or run "jupyter notebook IFCNN_Notebook.ipynb" to run the code part by part


### Highlights:
- Propose a general image fusion framework based on convolutional neural network
- Demonstrate good generalization ability for fusing various types of images
- Perform comparably or even better than other algorithms on four image datasets
- Create a large-scale and diverse multi-focus image dataset for training CNN models
- Incorporate perceptual loss to boost the model’s performance



### Architecture of our image fusion model:
![flowchart](https://github.com/uzeful/IFCNN/blob/master/flowchart.png)



### Datasets:
1. Multi-focus image dataset: [Datasets/CMFDataset](https://github.com/uzeful/IFCNN/blob/master/Code/datasets/CMFDataset)
2. Infrared and visual image dataset: [Datasets/IVDataset](https://github.com/uzeful/IFCNN/blob/master/Code/datasets/IVDataset)
3. Multi-modal medical image dataset: [Datasets/MDDataset](https://github.com/uzeful/IFCNN/blob/master/Code/datasets/MDDataset)
4. Multi-exposure image dataset: [Datasets/MEDataset](https://github.com/uzeful/IFCNN/blob/master/Code/datasets/MEDataset)



### Comparison Examples:
1. Multi-focus image fusion
![CMF05](https://github.com/uzeful/IFCNN/blob/master/Comparisons/CMF05.png)


2. Infrared and visual image fusion
![CMF05](https://github.com/uzeful/IFCNN/blob/master/Comparisons/IVroad.png)


3. Multi-modal medical image fusion
![MDc02](https://github.com/uzeful/IFCNN/blob/master/Comparisons/MDc02.png)


4. Multi-exposure image fusion
![MEdoor](https://github.com/uzeful/IFCNN/blob/master/Comparisons/MEdoor.png)



### Other Results of Our Model:
1. Multi-focus image dataset: [Results/CMF](https://github.com/uzeful/IFCNN/tree/master/Results/CMF)
2. Infrared and visual image dataset: [Results/IV](https://github.com/uzeful/IFCNN/tree/master/Results/IV)
3. Multi-modal medical image dataset: [Results/MD](https://github.com/uzeful/IFCNN/tree/master/Results/MDDataset)
4. Multi-exposure image dataset: [Results/ME](https://github.com/uzeful/IFCNN/tree/master/Results/ME)

### Typos
Eq. (4) in our paper is wrongly written, please the correct expression can be found from the official expression in [OpenCV document](https://docs.opencv.org/3.4.2/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa), i.e.,
![Eq4](https://github.com/uzeful/IFCNN/blob/master/Others/Eq4.png), where i=0...2kr, \alpha is the scale factor chosen so that \sum G(i)=1, ksize=2kr+1 and \sigma=0.6(ksize−1)+0.8.


### Citation:
If you find this code is useful for your research, please consider to cite our paper.
```
@article{zhang2019IFCNN,
  title={IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network},
  author={Zhang, Yu and Liu, Yu and Sun, Peng and Yan, Han and Zhao, Xiaolin and Zhang, Li},
  journal={Information Fusion},
  volume={54},
  pages={99--118},
  year={2020},
  publisher={Elsevier}
}
```