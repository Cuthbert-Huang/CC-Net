# Complementary consistency semi-supervised learning for 3D left atrial image segmentation
by Hejun Huang, Zuguo Chen*, Chaoyang Chen, Ming Lu, Ying Zou
## Introduction
This repository is for our paper '[Complementary consistency semi-supervised learning for 3D left atrial image segmentation](https://arxiv.org/abs/2210.01438)'.
## Requirements
This repository is based on Pytorch 1.9.1, CUDA11.1 and Python 3.6.5
## Usage
### Install
Clone the repo:
```shell
git clone https://github.com/Cuthbert-Huang/CC-Net.git 
```
### Dataset
We use [the dataset of 2018 Atrial Segmentation Challenge](http://atriaseg2018.cardiacatlas.org/). The processed h5 datas were provided in [googleDrive](https://drive.google.com/drive/folders/15Z2gmJCZuLbOYjX5RIlmpAxZYKMib1kI?usp=sharing) and [baiduNetdisk](https://pan.baidu.com/s/1WN4DKsrx-830KcT89pWiLg) (password: cuth). Please unzip and put them in the `data/LA` folder.
### Preprocess
If you want to process .nrrd data into .h5 data, you can use `code/dataloaders/preprocess.py`.
### Pretrained models
The pretrained models were provided in [googleDrive](https://drive.google.com/drive/folders/19qymbWUnjcBT_Cu3SmvmBhKAnRSKyq1i?usp=sharing) and [baiduNetdisk](https://pan.baidu.com/s/1LK42sJSJTMrgBG6JdqND5Q) (password: cuth).
Please put them in the `pretrained` folder.
### Train
If you want to train CC-Net for 10% labels on LA.
```shell
cd CC-Net
python ./code/train_ccnet_3d_v1.py --dataset_name LA --model ccnet3d_v1 --exp CCNET --labelnum 8 --gpu 0 --temperature 0.1 --max_iteration 10000
```
If you want to train CC-Net for 20% labels on LA.
```shell
cd CC-Net
python ./code/train_ccnet_3d_v1.py --dataset_name LA --model ccnet3d_v1 --exp CCNET --labelnum 16 --gpu 0 --temperature 0.1 --max_iteration 10000
```
### Test
If you want to test CC-Net for 10% labels on LA.
```shell
cd CC-Net
python ./code/test.py --dataset_name LA --model ccnet3d_v1 --exp CCNET --labelnum 8 --gpu 0
```
If you want to test CC-Net for 20% labels on LA.
```shell
cd CC-Net
python ./code/test.py --dataset_name LA --model ccnet3d_v1 --exp CCNET --labelnum 16 --gpu 0
```
## Citation
If our CC-Net is useful for your research, please consider citing:
```
@article{huang2022complementary,
  title={Complementary consistency semi-supervised learning for 3D left atrial image segmentation},
  author={Huang, Hejun and Chen, Zuguo and Chen, Chaoyang and Lu, Ming and Zou, Ying},
  journal={arXiv preprint arXiv:2210.01438},
  year={2022}
}
```
If you use the dataset of 2018 Atrial Segmentation Challenge, please consider citing:
```
@article{Xiong_A_global2021,
  author = {Xiong, Zhaohan and Xia, Qing and Hu, Zhiqiang and Huang, Ning and Bian, Cheng and Zheng, Yefeng and Vesal,          Sulaiman and Ravikumar, Nishant and Maier, Andreas and Yang, Xin},
  title = {A global benchmark of algorithms for segmenting the left atrium from late gadolinium-enhanced cardiac magnetic resonance imaging},
  journal = {Medical Image Analysis},
  year = {2021} }
```
## Acknowledgements
Our code is origin from [UAMT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [DTC](https://github.com/HiLab-git/DTC), and [MC-Net+](https://github.com/ycwu1997/MC-Net). Thanks to these authors for their excellent work.
