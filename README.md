# Deformation and Correspondence Aware Unsupervised Synthetic-to-Real Scene Flow Estimation for Point Clouds
This is the code related to "Deformation and Correspondence Aware Unsupervised Synthetic-to-Real Scene Flow Estimation for Point Clouds" (CVPR 2022) [[arXiv]](https://arxiv.org/pdf/2203.16895.pdf) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Jin_Deformation_and_Correspondence_Aware_Unsupervised_Synthetic-to-Real_Scene_Flow_Estimation_for_CVPR_2022_paper.pdf)
<p align='center'>
  <img src='network.png' width="1000px">
</p>


## GTA-V Scene Flow (GTA-SF) Dataset
[[Download link](https://1drv.ms/u/s!Ap1U6ygZ8oBwhCJgydLCFJpfZyFD?e=6G4ngc)]

GTA-SF is a large-scale synthetic dataset for real-world scene flow estimation. It contains 54,287 pairs of consecutive point clouds with densely annotated scene flow.
Some examples of our generated data (Ground Truth refers to the result of adding scene flow to the first frame):
<p align='center'>
  <img src='GTA-SF.png' width="1000px">
</p>

## Environment
* Python 3.6
* Pytorch 1.8.0
* CUDA 11.1

### Setup
```
conda create -n DCA_SRSFE python=3.6
conda activate DCA_SRSFE
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install numba=0.38.1
pip install cffi
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
cd ../..
```

## Data preparation
### GTA-SF
Download the [GTA-SF dataset](https://1drv.ms/u/s!Ap1U6ygZ8oBwhCJgydLCFJpfZyFD?e=6G4ngc) and organize all files as follows:
```
|—— data
|   |── GTA-SF
|   |   |── 00
|   |   |—— 01
|   |   |...
|   |   |—— 05
```

### Waymo
Please follow the paper [Scalable Scene Flow from Point Clouds in the Real World](https://arxiv.org/pdf/2103.01306.pdf) and go to [Waymo Open Dataset](https://waymo.com/open/download/) for registration, then download [scene flow labels](https://console.cloud.google.com/storage/browser/waymo_open_dataset_scene_flow).
1. Process raw data into point clouds. We use the same way as [ST3D](https://github.com/CVMI-Lab/ST3D/blob/master/docs/GETTING_STARTED.md#waymo-open-dataset). It is recommended to create a new conda environment to process data.
2. Copy files under DCA-SRSFE/data_preprocessing/Waymo/ to /ST3D/pcdet/datasets/waymo/. Extract scene flow labels by running: 
``` 
python generate_flow.py --train_dir TRAIN_DIR --valid_dir VALID_DIR 
```
3. Pack point clouds and scene flow labels into .npz files for training: ``` python save_npz.py --root_path ROOT_PATH ```

### Lyft
1. Download the official [Lyft Perception Dataset](https://level-5.global/data/perception/) training dataset. Then unpack train.tar and put them into the trainval folder.
2. Copy files under DCA-SRSFE/data_preprocessing/Lyft/ to /ST3D/pcdet/datasets/lyft/.
3. Install lyft_dataset_sdk before process lyft data: 
``` 
pip install -U lyft_dataset_sdk
``` 
* We modified the source codes (lyftdataset.py and data_classes.py) of lyft_dataset_sdk to include 'instance_token' in Box class. The modified files can be found in ./lyft_dataset_sdk. You can download them and replace the original files of lyft_dataset_sdk.
run lyft_process:
``` 
python lyft_process.py --save_dir SAVE_PATH --root_path ROOT_PATH 
```

### KITTI Scene Flow and FlyingThings3D
We follow [HPLFlowNet](https://github.com/laoreja/HPLFlowNet) to preprocess [KITTI Scene Flow 2015](http://www.cvlibs.net/download.php?file=data_scene_flow.zip) and FlyingThings3D datasets. Download and unzip it to 


After data preprocessing, they should be organized as follows:
```
|—— data
|   |── FlyingThings3D_subset_processed_35m
|   |── GTA-SF
|   |── KITTI_processed_occ_final
|   |── Lyft
|   |── waymo
```

## Usage
Take GTA-SF->Waymo as an example, you can download the [model](https://1drv.ms/u/s!Ap1U6ygZ8oBwhBSpZjaMHM4CbrJM?e=Jw5khg) pretrained on GTA-SF and run the following code for domain adaptation:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 1234 main_DCAdapt.py configs/train_adapt_gta_waymo.yaml
```
You can change the config file (.yaml) to switch the datasets and test the trained models.

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

## Citation
If you find it helpful to your research, please cite as follows:
```
@InProceedings{Jin_2022_CVPR,
    author    = {Jin, Zhao and Lei, Yinjie and Akhtar, Naveed and Li, Haifeng and Hayat, Munawar},
    title     = {Deformation and Correspondence Aware Unsupervised Synthetic-to-Real Scene Flow Estimation for Point Clouds},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {7233-7243}
}
```

## Acknowledgments
Our codes are based on [HPLFlowNet](https://github.com/laoreja/HPLFlowNet). The deformation regularization implementation is based on [Rigid3DSceneFlow](https://github.com/zgojcic/Rigid3DSceneFlow). The data processing codes for Waymo and Lyft are based on [ST3D](https://github.com/CVMI-Lab/ST3D) and [OpenPCDet v0.2](https://github.com/open-mmlab/OpenPCDet/tree/v0.2.0).
