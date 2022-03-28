# Deformation and Correspondence Aware Unsupervised Synthetic-to-Real Scene Flow Estimation for Point Clouds
This is the code related to "Deformation and Correspondence Aware Unsupervised Synthetic-to-Real Scene Flow Estimation for Point Clouds" (CVPR 2022).
<p align='center'>
  <img src='network.png' width="1000px">
</p>


## GTA-V Scene Flow (GTA-SF) Dataset
[Download link](https://1drv.ms/u/s!Ap1U6ygZ8oBwhCJgydLCFJpfZyFD?e=6G4ngc)

Some examples of our generated data (Ground Truth refers to the result of adding scene flow to the first frame):
<p align='center'>
  <img src='GTA-SF.png' width="1000px">
</p>

## Environment
* Python 3.6
* Pytorch 1.8.0
* CUDA 11.1

## Data preparation
### GTA-SF
Download the [GTA-SF dataset](https://1drv.ms/u/s!Ap1U6ygZ8oBwhCJgydLCFJpfZyFD?e=6G4ngc) and organize all files as follows:
```
|—— dataset
|   |── GTA-SF
|   |   |── 00
|   |   |—— 01
|   |   |...
|   |   |—— 05
```

### Waymo
Please follow the paper [Scalable Scene Flow from Point Clouds in the Real World](https://arxiv.org/pdf/2103.01306.pdf) and go to [Waymo Open Dataset](https://waymo.com/open/download/), then download [scene flow labels](https://pantheon.corp.google.com/storage/browser/waymo_open_dataset_scene_flow).


### Lyft
Download the official [Lyft Perception Dataset](https://level-5.global/data/perception/),

### KITTI Scene Flow
We follow [HPLFlowNet](https://github.com/laoreja/HPLFlowNet) to preprocess [KITTI Scene Flow 2015](http://www.cvlibs.net/download.php?file=data_scene_flow.zip). Download and unzip it to 

## Get started
### Setup
```
conda create -n DCA_SRSFE python=3.6
conda activate DCA_SRSFE
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install numba==0.38.1
pip install ciffi
pip install pyyaml==5.3.1
pip install tqdm
pip install scikit-learn==0.24.2

```
Setup for [HPLFlowNet](https://github.com/laoreja/HPLFlowNet) and [Pointnet2.Pytorch](https://github.com/sshaoshuai/Pointnet2.PyTorch):
```
cd models
python build_khash_cffi.py
cd pointnet2
python setup.py install
cd ../
```

### Pretrained models
Models pretrained on source domain and our checkpoints of synthetic-to-real domain adaptation can be found [here](https://1drv.ms/u/s!Ap1U6ygZ8oBwhBSpZjaMHM4CbrJM?e=Jw5khg). They should be organized as follows:
```
|—— checkpoints
|   |── trained_models
|   |   |── GTA_pretrained
|   |   |—— FT3D_pretrained
|   |   |—— to_Waymo
|   |   |—— to_Lyft
|   |   |—— to_KITTI
```

### Inference
coming soon...

## Citation
