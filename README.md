# EIA-MVS

## Adaptive Feature Enhanced Multi-View Stereo with Epipolar Line Information Aggregation (RAL)

## [Paper](https://ieeexplore.ieee.org/abstract/document/10700633)


## Installation

```bash
conda create -n eiamvs python=3.7.9
conda activate etmvsnet
pip install -r requirements.txt
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Data Preparation

#### 1. DTU Dataset

**Training data**. We use the same DTU training data as mentioned in MVSNet and CasMVSNet. Download [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) and [Depth raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip). Unzip and organize them as:
```
dtu_training                     
    ├── Cameras                
    ├── Depths   
    ├── Depths_raw
    └── Rectified
```


**Testing Data**. Download [DTU testing data](https://drive.google.com/file/d/135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_/view). Unzip it as:
```
dtu_testing                                       
    ├── scan1   
    ├── scan4
    ├── ...
```

#### 2. BlendedMVS Dataset

Download [BlendedMVS](https://drive.google.com/file/d/1ilxls-VJNvJnB7IaFj7P0ehMPr7ikRCb/view) and unzip it as:

```
blendedmvs                          
    ├── 5a0271884e62597cdee0d0eb                
    ├── 5a3ca9cb270f0e3f14d0eddb   
    ├── ...
    ├── training_list.txt
    ├── ...
```

#### 3. Tanks and Temples Dataset

Download [Tanks and Temples](https://drive.google.com/file/d/1YArOJaX9WVLJh4757uE8AEREYkgszrCo/view) and  unzip it as:
```
tanksandtemples                          
       ├── advanced                 
       │   ├── Auditorium       
       │   ├── ...  
       └── intermediate
           ├── Family       
           ├── ...          
```
We use the camera parameters of short depth range version (included in your download), you should replace the `cams` folder in `intermediate` folder with the short depth range version manually.



## Training

### Training on DTU

To train the model from scratch on DTU, specify ``DTU_TRAINING`` in ``train_dtu.sh`` first and then run:
```
bash train_dtu.sh
```

### Finetune on BlendedMVS

To fine-tune the model on BlendedMVS, you need specify `BLD_TRAINING` and `BLD_CKPT_FILE` in `train_bld.sh` first, then run:
```
bash train_bld.sh
```


## Testing

### Testing on DTU

For DTU testing, we use the model ([pretrained model](https://drive.google.com/file/d/1D82VEWimgnB_cDwrUGJaIZTEXU0v--Xf/view?usp=drive_link)) trained on DTU training dataset. Specify `DTU_TESTPATH` and `DTU_CKPT_FILE` in `test_dtu.sh` first, then run the following command to generate point cloud results.
```
bash test_dtu.sh
```
For quantitative evaluation, download [SampleSet](http://roboimagedata.compute.dtu.dk/?page_id=36) and [Points](http://roboimagedata.compute.dtu.dk/?page_id=36) from DTU's website. Unzip them and place `Points` folder in `SampleSet/MVS Data/`. The structure is just like:
```
SampleSet
├──MVS Data
      └──Points
```

Specify `datapath`, `plyPath`, `resultsPath` in `evaluations/dtu/BaseEvalMain_web.m` and `datapath`, `resultsPath` in `evaluations/dtu/ComputeStat_web.m`, then run the following command to obtain the quantitative metics.
```
cd evaluations/dtu
matlab -nodisplay
BaseEvalMain_web 
ComputeStat_web
```

### Testing on Tanks and Temples
We recommend using the finetuned model ([pretrained model](https://drive.google.com/file/d/1T-e09S5Dd6I0x9CATPHYV143L-w8NEsF/view?usp=drive_link)) to test on Tanks and Temples benchmark. Similarly, specify `TNT_TESTPATH` and `TNT_CKPT_FILE` in `test_tnt_inter.sh` and `test_tnt_adv.sh`. To generate point cloud results, just run:
```
bash test_tnt_inter.sh
```
```
bash test_tnt_adv.sh
``` 
For quantitative evaluation, you can upload your point clouds to [Tanks and Temples benchmark](https://www.tanksandtemples.org/).

## Citation
```bibtex
@ARTICLE{wang2024eiamvs,
  author={Wang, Shaoqian and Li, Bo and Yang, Jian and Dai, Yuchao},
  journal={IEEE Robotics and Automation Letters}, 
  title={Adaptive Feature Enhanced Multi-View Stereo With Epipolar Line Information Aggregation}, 
  year={2024},
  volume={9},
  number={11},
  pages={10439-10446}}
```


## Acknowledgements
Our work is partially based on these opening source work: [MVSNet](https://github.com/YoYo000/MVSNet), [cascade-stereo](https://github.com/alibaba/cascade-stereo), [ET-MVSNet](https://github.com/TQTQliu/ET-MVSNet).

We appreciate their contributions to the MVS community.
