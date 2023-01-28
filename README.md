# RFNet
The codes for 3QNet: 3D Point Cloud Geometry Quantization Compression Network published in ACM TRANSACTIONS ON GRAPHICS.

## Environment
* TensorFlow 1.13.1
* Cuda 10.0
* Python 3.6.9
* numpy 1.14.5
* Sklearn
* Draco
* h5py

## Dataset
The adopted object models, indoor scenes, and outdoor scenes can be found in [Visionair](https://github.com/yulequan/PU-Net), [Scannet](https://github.com/charlesq34/pointnet2), [Kitti](https://github.com/PRBonn/semantic-kitti-api).

## Usage

1. Compile

```
cd ./tf_ops
bash compile.sh
```

2. Train

```
Python3 3qnet.py
```
Note that the training h5 data should be put under `./data`. The path for saved checkpoint files(`savepath`) should be edited according to your setting.

3. Test

```
Python3 test3q.py
```
Please put the ply files to be processed in `indir`, set the `dracodir` as where you install Draco, and edit `modelpath` as the path of checkpoint. The point clouds will be encoded into binary code files in `bindir` and decompressed into point clouds again into `outdir`. Note the compression level could be adjusted by changing `level` from 1 to 8.
Some decompressed samples under about 1bpp could be
![image](https://github.com/Tianxinhuang/3QNet/blob/master/tog_quali.jpg)

## Citation
If you find our work useful for your research, please cite:
```
@article{huang20223qnet,
  title={3QNet: 3D Point Cloud Geometry Quantization Compression Network},
  author={Huang, Tianxin and Zhang, Jiangning and Chen, Jun and Ding, Zhonggan and Tai, Ying and Zhang, Zhenyu and Wang, Chengjie and Liu, Yong},
  journal={ACM Transactions on Graphics (TOG)},
  volume={41},
  number={6},
  pages={1--13},
  year={2022},
  publisher={ACM New York, NY, USA}
}
```
