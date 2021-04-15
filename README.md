# Efficient Space-time Video Super Resolution using Low-Resolution Flow and Mask Upsampling

_Saikat Dutta, Nisarg A. Shah, Anurag Mittal_

Accepted at NTIRE workshop, collocated with CVPR 2021 [ArXiv](https://arxiv.org/abs/2104.05778)

### Requirements
Create a conda environment with Pytorch-1.1, CuPy-6.0, OpenCV, SciPy.
```
conda create -n myenv
conda activate myenv
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
conda install -c anaconda cupy
conda install -c anaconda scipy
conda install -c conda-forge opencv
```
### Dataset
We use REDS STSR dataset for training and validation. Get the dataset by registering [here](https://competitions.codalab.org/competitions/28072#learn_the_details).
Unzip the dataset under `REDS/` directory.
```
---REDS/
  |---val/
    |---val_sharp_bicubic/
      |---X4/
    |---val_sharp/
    
```

### Generate results on REDS STSR Validation dataset
```
python REDS_val.py
```

### Acknowledgement
The following repositories were used to develop this project :

[1] [QVI](https://sites.google.com/view/xiangyuxu/qvi_nips19)

[2] [RSDN](https://github.com/junpan19/RSDN)

[3] [PWCNet](https://github.com/sniklaus/pytorch-pwc)
