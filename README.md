# IFCNN
Project page of  "IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network,  Information Fusion, Under Revision". 



Highlights :

- Propose a general image fusion framework based on convolutional neural network
- Demonstrate good generalization ability for fusing various types of images
- Perform comparably or even better than other algorithms on four image datasets
- Create a large-scale and diverse multi-focus image dataset for training CNN models
- Incorporate perceptual loss to boost the model’s performance



The structure of our image fusion model is as follows.

![flowchart](.\flowchart.png)



Datasets:

1. Multi-focus image dataset: [CMFDataset](.\Datasets\CMFDataset )
2. Infrared and visual image dataset: [IVDataset](.\Datasets\IVDataset)
3. Multi-modal medical image dataset: [MDDataset](.\Datasets\MDDataset)
4. Multi-exposure image dataset: [MEDataset](.\Datasets\MEDataset)



Comparison Examples:

![CMF05](.\Comparisons\CMF05.png)



![CMF05](.\Comparisons\IVroad.png)



![MDc02](.\Comparisons\MDc02.png)



![MEdoor](.\Comparisons\MEdoor.png)



Other Results of Our Model:

1. Multi-focus image dataset: [Results\CMF](.\Results\CMF )
2. Infrared and visual image dataset: [Results\IV](.\Results\IV)
3. Multi-modal medical image dataset: [Results\MD](.\Results\MDDataset)
4. Multi-exposure image dataset: [Results\ME](Results\ME)