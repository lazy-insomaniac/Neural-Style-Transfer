# Extreme Low Light Image Denoising  
# What it does ❓
The aim of the project was to build a model that allows you to take images and 
build new images with different artistic styles. So, in order to accomplish this 
task, using the research paper A Neural Algorithm of Artistic Style, we 
implemented our own version of Neural style transfer, that generates well 
balanced images having good representation of content and style, we use VGG 
19 pretrained network to compute losses and apply optimizer on image itself.  
# EXAMPLE 
![image](https://github.com/lazy-insomaniac/Neural-Style-Transfer/assets/114395022/7f4975c0-a5cc-4e2a-893c-d510c4fbca7e)
# Description 📝
VGG19 is a variant of the VGG model which in short consists of 19 layers (16 
convolution layers, 3 Fully connected layer, 5 MaxPool layers and 1 SoftMax 
layer). It was trained on millions of images, as a result, the network learnt rich 
feature representations for a wide range of images, this allows us to use it 
perform different tasks using it, with our focus primarily on Neural style 
transfer.  We set all the RELU layers as inplace = False, because they cause 
problems for content loss function discussed ahead.
The main.py file for my code can be run directly after setting the respective paths for content and style image or you can use the notebook on Kaggle by importing it and setting path accordingly.
This study primarily focuses on implementing a version of research paper A Neural Algorithm of Artistic Style.
-	Obtain high quality style images   
-	Obtain images with balance between style and content for any images given.

# Download Links: 🔗
https://drive.google.com/file/d/1Ov7xwG9VEw6F1JLK3MOKL8BC4GuCNCWu/view?usp=drive_link 

* Neural Style Transfer dataset📊 : https://www.kaggle.com/datasets/burhanuddinlatsaheb/neural-styletransfer/data
* Dataset Description:
- A small dataset named Neural Style Transfer on Kaggle was used. It contains 14 
images in total  
  - Statistics
     - Content – 7 images  
     - Style – 7 images

# Installation 🔧
  - Requirements:
    - Python 
    - Torch 
    - TorchVision
    -  cuda : 11.8 (P100 and T4x2 GPU)(for Training)
    -  NumPy: 1.16.4 or higher
    - VGG 19
    - PIL

# Setup ⚙️
  - pip install tensorflow
  - pip install torch torchvision


# Documentation 📑
 - HLD: Project_Report: [https://drive.google.com/file/d/1Ov7xwG9VEw6F1JLK3MOKL8BC4GuCNCWu/view?usp=drive_link](https://drive.google.com/file/d/1Bg57Z38PaLd2TcANpoFvaHQpaugCrC6P/view?usp=drive_link)

# References ⚓
 - A Neural Algorithm of Artistic Style : https://arxiv.org/pdf/1508.06576
 - Neural Transfer using Pytorch : https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
