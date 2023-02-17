## 第一步
1.解压项目代码，项目目录如下

```
lanenet-lane-detection-master
├─ .idea
│  └─ vcs.xml
├─ config
│  └─ tusimple_lanenet.yaml
├─ data
│  ├─ source_image
│  │  ├─ accuracy.png
│  │  ├─ binary_seg_loss.png
│  │  ├─ instance_seg_loss.png
│  │  ├─ lanenet_batch_test.gif
│  │  ├─ lanenet_binary_seg.png
│  │  ├─ lanenet_embedding.png
│  │  ├─ lanenet_instance_seg.png
│  │  ├─ lanenet_mask_result.png
│  │  ├─ network_architecture.png
│  │  ├─ qr.jpg
│  │  └─ total_loss.png
│  ├─ training_data_example
│  │  ├─ gt_binary_image
│  │  │  ├─ 0000.png
│  │  │  ├─ 0001.png
│  │  │  ├─ 0002.png
│  │  │  ├─ 0003.png
│  │  │  ├─ 0004.png
│  │  │  └─ 0005.png
│  │  ├─ gt_instance_image
│  │  │  ├─ 0000.png
│  │  │  ├─ 0001.png
│  │  │  ├─ 0002.png
│  │  │  ├─ 0003.png
│  │  │  ├─ 0004.png
│  │  │  └─ 0005.png
│  │  ├─ image
│  │  │  ├─ 0000.png
│  │  │  ├─ 0001.png
│  │  │  ├─ 0002.png
│  │  │  ├─ 0003.png
│  │  │  ├─ 0004.png
│  │  │  └─ 0005.png
│  │  ├─ train.txt
│  │  └─ val.txt
│  ├─ tusimple_ipm_remap.yml
│  └─ tusimple_test_image
│     ├─ 0.jpg
│     ├─ 1.jpg
│     ├─ 2.jpg
│     └─ 3.jpg
├─ data_provider
│  ├─ lanenet_data_feed_pipline.py
│  └─ tf_io_pipline_tools.py
├─ lanenet_model
│  ├─ lanenet.py
│  ├─ lanenet_back_end.py
│  ├─ lanenet_discriminative_loss.py
│  ├─ lanenet_front_end.py
│  ├─ lanenet_postprocess.py
│  └─ __init__.py
├─ LICENSE
├─ local_utils
│  ├─ config_utils
│  │  ├─ parse_config_utils.py
│  │  └─ __init__.py
│  └─ log_util
│     ├─ init_logger.py
│     └─ __init__.py
├─ mnn_project
│  ├─ config.ini
│  ├─ config_parser.cpp
│  ├─ config_parser.h
│  ├─ convert_lanenet_model_into_mnn_model.sh
│  ├─ dbscan.hpp
│  ├─ freeze_lanenet_model.py
│  ├─ kdtree.cpp
│  ├─ kdtree.h
│  ├─ lanenet_model.cpp
│  ├─ lanenet_model.h
│  └─ __init__.py
├─ README.md
├─ requirements.txt
├─ semantic_segmentation_zoo
│  ├─ bisenet_v2.py
│  ├─ cnn_basenet.py
│  ├─ vgg16_based_fcn.py
│  └─ __init__.py
├─ tools
│  ├─ evaluate_lanenet_on_tusimple.py
│  ├─ evaluate_model_utils.py
│  ├─ generate_tusimple_dataset.py
│  ├─ make_tusimple_tfrecords.py
│  ├─ test_lanenet.py
│  └─ train_lanenet_tusimple.py
├─ trainner
│  ├─ tusimple_lanenet_multi_gpu_trainner.py
│  ├─ tusimple_lanenet_single_gpu_trainner.py
│  └─ __init__.py
└─ _config.yml
 
```

## 2.解压数据集到data文件夹新建文件夹tuSimple内

```
tuSimple/
├── clips
│   ├── 0313-1
│   ├── 0313-2
│   ├── 0530
│   ├── 0531
│   └── 0601
├── label_data_0313.json
├── label_data_0531.json
├── label_data_0601.json
├── readme.md
└── test_tasks_0627.json
```

## 3.安装必要的环境
### 3.1 安装conda、pycharm和CUDA

安装anaconda(安装过程记得添加环境变量)
https://www.anaconda.com/download/

安装pycharm，一个非常好用的python开发环境。(社区版)
https://www.jetbrains.com/pycharm/download/download-thanks.html?platform=windows&code=PCC

CUDA下载网址(v10.0)
https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_411.31_win10

双击安装先会安装一个安装程序。安装完成后会自动添加环境变量。

cuDNN下载网址(v7.4.2)
https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.4.2/prod/10.0_20181213/cudnn-10.0-windows10-x64-v7.4.2.24.zip

cuDNN操作
将cudnn目录下的文件移入到对应的cuda文件夹中
![image](https://user-images.githubusercontent.com/36353749/219566381-2aef6455-94ae-4b4c-b772-d446f07ea4f1.png)

### 3.2 创建conda环境
```
conda create -n py36 python==3.6
conda activate py36
```
然后配置好pycharm项目的python解释器。
### 3.3 安装环境必要的软件包
```
pip install -r requirements.txt
```

## 4. 生成用于训练的数据
### 4.1 生成training和testing文件夹
```
#脚本位于tools目录下
python generate_tusimple_dataset.py --src_dir=YOURPATH\data\tuSimple
```

几分钟后，在tuSimple目录下自动生成了training和testing两个目录，如下所示：
```
tuSimple/
|
├──training/
|   ├── gt_binary_image
|   ├── gt_image
|   ├── gt_instance_image
|   ├── label_data_0313.json
|   ├── label_data_0531.json
|   ├── label_data_0601.json
|   └── train.txt
└──testing/
    └── test_tasks_0627.json
```
### 4.2 分割出测试集
**train.txt:** 用来训练的图片文件的文件名列表 （训练集）2186
**val.txt:** 用来验证的图片文件的文件名列表 （验证集）720
**test.txt** 用来测试的图片文件的文件名列表 （测试集）720
![image](https://user-images.githubusercontent.com/36353749/219576755-ab4573a4-19f3-4f33-aa56-69d390fa9a1b.png)

该脚本仅生成了train.txt，我们可以手动分割一下train set和val set，也就是剪切train.txt中的一部分到一个新建的val.txt和test.txt文件中。

### 4.3 修改配置文件config/tusimple_lanenet.yaml
路径修改
```
DATASET:
    DATA_DIR: 'YOURPATH\data\tuSimple\training'
    IMAGE_TYPE: 'rgb'  # choice rgb or rgba
    NUM_CLASSES: 2
    TEST_FILE_LIST: 'YOURPATH\data\tuSimple\training\test.txt'
    TRAIN_FILE_LIST: 'YOURPATH\data\tuSimple\training\train.txt'
    VAL_FILE_LIST: 'YOURPATH\data\tuSimple\training\val.txt'
```
根据显卡配置修改batchsize(8)
```
TRAIN:
    MODEL_SAVE_DIR: 'model/tusimple/'
    TBOARD_SAVE_DIR: 'tboard/tusimple/'
    MODEL_PARAMS_CONFIG_FILE_NAME: "model_train_config.json"
    RESTORE_FROM_SNAPSHOT:
        ENABLE: False
        SNAPSHOT_PATH: ''
    SNAPSHOT_EPOCH: 8
    BATCH_SIZE: 8
```
### 4.4 将标注文件转换为tensorflow的record

首先编辑make_tusimple_tfrecords.py文件，插入

```
#脚本位于tools目录下
python make_tusimple_tfrecords.py --dataset_dir YOURPATH\data\tuSimple\training --tfrecords_dir YOURPATH\data\tuSimple\training\tfrecords
```
等待几分钟，脚本会在项目的data/training/tfrecords目录下生成相应的tfrecord文件

## 5. 训练
```
#脚本位于tools目录下
python train_lanenet_tusimple.py
```
或
```
python train_lanenet_tusimple.py --dataset_dir YOURPATH\data\tuSimple\training --multi_gpus False
```
tensorboar查看训练过程
```
cd tboard/tusimple/bisenetv2_lanenet
tensorboard --logdir=.
```
tensorboard查看模型在验证集上的 **总损失(val_cost)、分割损失（val_binary_seg_loss）、嵌入损失（val_instance_seg_loss）以及分割精度（val_accuracy）** 变化曲线
