# 算能镜像下载文档（Ubuntu系统）
## 一、	Ubuntu国内镜像下载地址
1.中科大：http://mirrors.ustc.edu.cn/ubuntu-releases/

2.清华源：https://mirrors.tuna.tsinghua.edu.cn/ubuntu-releases/

## 二、	TPU开发环境配置
### 1.开发环境配置
        首先检查当前系统环境是否满足ubuntu 22.04和python 3.10。如不满足，请进行下一节 基础环境配置 ；如满足，直接跳至 tpu_mlir 安装 。
### 1.1基础环境配置
        如不满足上述系统环境，则需要使用Docker。首次使用Docker, 可执行下述命令进行安装和配置（仅首次执行，非首次跳过）：
```bash
    # Ubuntu系统上首次安装docker使用命令
    sudo apt install docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo groupadd docker
    sudo usermod -aG docker $USER
    newgrp docker
```
### ①可从官网开发资料https://developer.sophgo.com/site/index/material/86/all.html 下载所需镜像文件，或使用下方命令下载镜像
```bash
    # 获取镜像文件
    wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/24/06/14/12/sophgo-tpuc_dev-v3.2_191a433358ad.tar.gz
    docker load -i sophgo-tpuc_dev-v3.2_191a433358ad.tar.gz
```
### ②需要确保镜像文件在当前目录，并在当前目录创建容器如下:
```bash
    # 创建容器
    mkdir workspace
    cd workspace
    docker run -it -d --name zzqy -v $PWD:/workspace -p 8002:8002 sophgo/tpuc_dev:v3.2 /bin/bash
```
其中， zzqy 为容器名称, 可以自定义； $PWD 为当前目录，与容器的 /workspace 目录同步。
#### ③启动容器， zzqy 为容器名称：
```bash
    docker start zzqy
    docker exec -it zzqy /bin/bash
```
#### ④安装tpu_mlir依赖：
```bash
    # 安装全部依赖
    pip install tpu_mlir[all]
```
### ⑤请从Github的 https://github.com/sophgo/tpu-mlir/releases/ 处下载tpu-mlir-resource.tar并解压，解压后将文件夹重命名为tpu_mlir_resource：
```bash
    # 安装全部依赖
    tar -xvf tpu-mlir-resource.tar
    mv regression/ tpu-mlir-resource/
```
## 三、	YOLOv8-seg模型编译(自定义模型)
### 1、YOLOv8-seg模型编译环境准备
### ①在tpu-mlir-resource/文件下
```bash
    cd tpu-mlir-resource/
```
### ②下载yolov8-seg编译文件和推理代码
    从Github的https://github.com/zgp20010421/YOLOv8-seg-Xray.git下载解压后,放在tpu-mlir-resource/文件下
```bash
    # git 出现网络问题，可直接点击链接，下载移到tpu-mlir-resource/文件下
    git clone https://github.com/zgp20010421/YOLOv8-seg-Xray.git
```
### ③启动容器， zzqy 为容器名称：
```bash
    # 退出容器 or重启 Ubuntu系统需要执行
    docker start zzqy
    docker exec -it zzqy /bin/bash  # 会直接进入到/workspace文件下
    cd tpu-mlir-resource/YOLOv8-seg-Xray  # 切换到YOLOv8-seg-Xray
```

### ④将自定义训练集，训练好的PT权重文件命名为yolov8s-seg-Xray-1b.pt,放在YOLOv8-seg-Xray/models/torch文件下
### ⑤创建YOLOv8-seg导出ONNX模型虚拟环境
```bash
    # 在YOLOv8-seg-Xray文件下
    python3 -m venv yolov8-seg  # 创建虚拟环境
    source yolov8_seg_venv/bin/activate  # 进入虚拟环境
    pip install ultralytics onnx -i https://pypi.tuna.tsinghua.edu.cn/simple # 下载所需环境
```
### 2、YOLOv8-seg模型导出和编译
### ①运行在python文件夹下的export.py（运行export.py之前需查看PT权重的路径和名称是否正确）
```bash
    cd python
    python3 export.py  # 导出为onnx模型，注意imgsz的大小(记录下来)
    mv yolov8s-seg-Xray-1b.onnx ../../onnx/ # 移动到onnx文件下
    deactivate # 退出虚拟环境
```
### ③编译Bmodel模型,在scripts文件下有编译为FP32、INT8Bmodel文件
```bash
    cd ../scripts/
    # 执行编译文件之前需要进行参数调整，对gen_fp32bmodel_mlir.py文件进行修改
    # 修改 gen_fp32bmodel_mlir('yolov8s', 'Xray', 1, '../models/bm1684', 'bm1684', 320, 'F32')函数参数 
    # yolov8s：模型名称, Xray：模型类别, 1：batch_size, ../models/bm1684: 输出文件路径, bm1684:芯片型号, 320:imgsz的大小, F32:编译数据类型 
    python3 gen_fp32bmodel_mlir.py # 编译为FP32Bmodel
    # 修改 gen_intbmodel_mlir('yolov8s', 'Xray', 1, '../models/bm1684', 'bm1684', 320, 'INT8')函数参数 
    python3 gen_int8bmodel_mlir.py # 编译为INT8Bmodel
```
### ④查看../models/bm1684文件是否存在相应的FP32、INT8Bmodel模型文件

### ⑤修改datasets文件下的test_Xray图片、coco_Xray.names（自定义数据集的img和labels）
```bash
    # coco_Xray.names的格式（labels的中英文）
    __background__:背景
    dental caries:龋齿
    impacted wisdom teeth:阻生智齿
    periapical disease:根尖周病
```
### 3、YOLOv8-seg文件打包到少林派(BM1684)
### ① 获取bm1684/文件下的yolov8s_Xray_F32_1b.bmodel的名称
```bash
    cd scripts/
    # vim mv_folder.sh 修改model_name="yolov8s_Xray_F32_1b"
    vim mv_folder.sh
    # 将需要的文件打包成以yolov8s_Xray_F32_1b命名的新文件夹
    sudo ./mv_folder.sh
```
### ② 检查yolov8s_Xray_F32_1b/models/bm1684是否存在bmodel模型，存在执行下述命令
```bash
    # 移植到少林派指令
    # 在docker环境中进行移植
    scp -r yolov8s_Xray_F32_1b/ linaro@192.168.10.1:/data
```

## 上述为全部导出、编译、打包、移植过程！😀😀😀
