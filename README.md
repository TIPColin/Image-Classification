# Image Classfication_NIvsCG
An reimplement code for paper "[Distinguishing Computer-Generated Images from Natural Images Using Channel and Pixel Correlation](https://link.springer.com/article/10.1007/s11390-020-0216-9)".
<div align=center>
  <img width="380" src="https://github.com/Evergrow/JCST_NIvsCG/blob/master/images/img1.jpg/" hspace="10">
  <img width="380" src="https://github.com/Evergrow/JCST_NIvsCG/blob/master/images/img2.jpg/" hspace="10">
</div>

Example results of our method on images from the highly challenging dataset of [Google versus PRCG](http://www.ee.columbia.edu/ln/dvmm/downloads/PIM_PRCG_dataset/). The demo lists the confidence of two labels (NI and CG) and calculates the time cost. 

## Prerequisites
* Ubuntu 16.04
* Python >=3.7 (>=3.8 for windows)
* clang （>=8.0 for mac）
* NVIDIA GPU CUDA 9.0 + cuDNN 7.1.4
* Jittor 
* Pillow (Python Imaging Library)
* Tqdm (Progress Bar for Python)

## Run
* Download the training and testing set: [Columbia PRCG](http://www.ee.columbia.edu/ln/dvmm/downloads/PIM_PRCG_dataset/) and [SPL2018](https://rose.ntu.edu.sg/Publications/Documents/Others/Computer%20Graphics%20Identification%20Combining%20Convolutional%20and%20Recurrent%20Neural%20Networks.pdf).
* Training: Set parameters in [train.py] or at the command line.
```python
python train.py --batch-size 64 --patch-size 96 --epoch 1200 --lr 0.001
```
* Testing: Run ```python test.py```.


## Acknowledgment
This code is an reimplement of code referring to [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and a Caffe version [NIvsCG](https://github.com/weizequan/NIvsCG).
