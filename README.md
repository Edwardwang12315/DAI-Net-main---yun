<p align="center">
  <h1 align="center">Boosting Object Detection with Zero-Shot Day-Night Domain Adaptation
</h1>
  <p align="center">
    <a href="https://zpdu.github.io/">Zhipeng Du</a>
    ·
    <a href="https://sites.google.com/site/miaojingshi/home">Miaojing Shi</a>
    ·
    <a href="https://jiankangdeng.github.io/">Jiankang Deng</a>
  </p>
  


PyTorch implementation of **Boosting Object Detection with Zero-Shot Day-Night Domain Adaptation**. (CVPR 2024) [[Page](https://zpdu.github.io/DAINet_page/) | [Paper](https://arxiv.org/abs/2312.01220)]

![overview](./assets/overview.png)



## 🔨 To-Do List

1. - [x] release the code regarding the proposed model and losses.
3. - [x] release the evaluation code, and the pretrained models.

3. - [x] release the training code.

## :rocket: Installation

Begin by cloning the repository and setting up the environment:

```
git clone https://github.com/ZPDu/DAI-Net.git
cd DAI-Net

conda create -y -n dainet python=3.7
conda activate dainet

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

## :notebook_with_decorative_cover: Training

#### Data and Weight Preparation

- Download the WIDER Face Training & Validation images at [WIDER FACE](http://shuoyang1213.me/WIDERFACE/).
- Obtain the annotations of [training set](https://github.com/daooshee/HLA-Face-Code/blob/main/train_code/dataset/wider_face_train.txt) and [validation set](https://github.com/daooshee/HLA-Face-Code/blob/main/train_code/dataset/wider_face_val.txt).
- Download the [pretrained weight](https://drive.google.com/file/d/1MaRK-VZmjBvkm79E1G77vFccb_9GWrfG/view?usp=drive_link) of Retinex Decomposition Net.
- Prepare the [pretrained weight](https://drive.google.com/file/d/1whV71K42YYduOPjTTljBL8CB-Qs4Np6U/view?usp=drive_link) of the base network.

Organize the folders as:

```
.
├── utils
├── weights
│   ├── decomp.pth
│   ├── vgg16_reducedfc.pth
├── dataset
│   ├── wider_face_train.txt
│   ├── wider_face_val.txt
│   ├── WiderFace
│   │   ├── WIDER_train
│   │   └── WIDER_val
```

#### Model Training

To train the model, run

```
python -m torch.distributed.launch --nproc_per_node=$NUM_OF_GPUS$ train.py
```

## :notebook: Evaluation​

On Dark Face:

- Download the testing samples from [UG2+ Challenge](https://codalab.lisn.upsaclay.fr/competitions/8494?secret_key=cae604ef-4bd6-4b3d-88d9-2df85f91ea1c).
- Download the checkpoints: [DarkFaceZSDA](https://drive.google.com/file/d/1BdkYLGo7PExJEMFEjh28OeLP4U1Zyx30/view?usp=drive_link) (28.0) or [DarkFaceFS](https://drive.google.com/file/d/1ykiyAaZPl-mQDg_lAclDktAJVi-WqQaC/view?usp=drive_link) (52.9, finetuned with full supervision).
- Set (1) the paths of testing samples & checkpoint, (2) whether to use a multi-scale strategy, and run test.py.
- Submit the results for benchmarking. ([Detailed instructions](https://codalab.lisn.upsaclay.fr/competitions/8494?secret_key=cae604ef-4bd6-4b3d-88d9-2df85f91ea1c)).

On ExDark:

- Our experiments are based on the codebase of [MAET](https://github.com/cuiziteng/ICCV_MAET). You only need to replace the checkpoint with [ours](https://drive.google.com/file/d/1g74-aRdQP0kkUe4OXnRZCHKqNgQILA6r/view?usp=drive_link) for evaluation.

# 调试记录
## 2025.1.22
- test输出只有预测txt文件，补充了把预测框绘制出来的步骤
- 简单筛选了一下，置信度小于0.3的不显示，效果很好
- 以上测试用的是作者提供的权重文件，只适用于人脸检测
- _C.TOP_K = 20时，mAP=14.19
- _C.TOP_K = 750时，mAP=14.21
## 2025.1.23
- 修改了部分网络结构

## 2025.3.20
- 损失函数仍然没法下降，感觉应该是金字塔校正的那部分初始化有问题
- 取消了deyolo部分的所有初始化 还是没法收敛
- 取消其他所有的网络，金字塔部分恢复到DENet原始结构，收敛慢但是正常收敛
- 增加正常亮度金字塔对比的高频损失，出现了分类和回归损失消失的问题
- 对DENet权重初始化
- 把多余的函数套用简化了
- 在代码里标注清楚，图像数据都是归一后的值
- 
- 修改了权重初始化函数，把kaiming初始化换成了原先的xavier初始化
- KL输入没有展平均衡化处理，在DENet中怎么让通道数增加。无法实现，DENet返回交换LF的重组图像：Light_dark,Dark_light
- 在VGG中对两幅图像进行初步提取特征再进行KL损失计算，出现nan的问题已解决
- CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py  训练指令      
- **临时启用鼠标**：在tmux会话中按下`Ctrl + b`，然后输入`:set -g mouse on`并按回车
- 损失函数显示错误, 修改为显示平均值，总loss应该显示当前100个iter的平均值(发现严重问题：损失一百次累加一次，次数计数无限累加。已修改，只显示近100个batch的总损失)
- 
- 修改后正确显示损失了，效果非常好（把mutual改成了5）
- 关机后也能正常运行
- val时显示loss不太准确

- DENet部分引入预训练权重，测试加速收敛效果如何：loss_mutual非常小
- 无预训练测试时长：3.25 日 20：00 至  日

- 增加动态亮度调整模块
- 增加细节亮度微调模块

```bash
# 安装 tmux（如果未安装）：
sudo apt-get install tmux  # Ubuntu/Debian
# 启动 tmux 会话：
tmux new -s training
# 在 tmux 会话中运行训练任务：
python train.py
# 分离会话（保持任务运行）：
按 Ctrl + B，然后按 D。
# 重新连接会话：
tmux attach -t training
```

## Github操作：

**创建仓库**

```bash
git init
git branch -M main # 建立仓库的main分支
git remote add origin git@github.com:**** # 和仓库建立远程联系
```

**提交修改**
```bash
git add -A # 添加所有文件到 缓存区
git commit -m "修改内容" # 缓存区内容提交到云区
git push --set-upstream origin main
```






## 📑 Citation

If you find this work useful, please cite

``` citation
@inproceedings{du2024boosting,
  title={Boosting Object Detection with Zero-Shot Day-Night Domain Adaptation},
  author={Du, Zhipeng and Shi, Miaojing and Deng, Jiankang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12666--12676},
  year={2024}
}
```

or

``` citation
@article{du2023boosting,
  title={Boosting Object Detection with Zero-Shot Day-Night Domain Adaptation},
  author={Du, Zhipeng and Shi, Miaojing and Deng, Jiankang},
  journal={arXiv preprint arXiv:2312.01220},
  year={2023}
}
```



## 🔎 Acknowledgement

We thank [DSFD.pytorch](https://github.com/yxlijun/DSFD.pytorch), [RetinexNet_PyTorch](https://github.com/aasharma90/RetinexNet_PyTorch), [MAET](https://github.com/cuiziteng/ICCV_MAET), [HLA-Face](https://github.com/daooshee/HLA-Face-Code) for their amazing works!

