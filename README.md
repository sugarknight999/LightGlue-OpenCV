# LightGlue-OpenCV 
LightGlue 是一种新的基于深度神经网络，用来匹配图像中的局部特征的深度匹配器。是 SuperGlue 的加强版本。相比于 SuperGlue，LightGlue 在内存和计算方面更高效，同时结果更准确，也更容易训练。
其**原项目地址**如下：
https://github.com/cvg/LightGlue

近期由于实习项目需要，需要在相机上进行实时的特征点追踪。然后配置了LightGlue 原项目。但是发现 LightGlue 计算推倒出图像上左右匹配的特征点后，在代码进行到的可视化环节时，会花费大量的时间，无法满足实习项目中实时性的要求。因此**在 viz2d.py 里对原项目可视化部分的代码进行修改，并添加了摄像头捕获图像后，根据捕获的第一帧图像的特征点进行特征匹配的 demo 文件 demo_camera.py**，便于下载者可以直接使用。
**且本项目保留了原有项目代码里的可视化部分 viz2d_bak.py，添加了一些注释便于后续读者阅读理解**。因为原项目里面并没有相关的 demo，因此本项目按照原本项目里面可视化的部分也编写了相关的 demo 文件 demo_bak.py、demo2image.py 可供学习参考。

## Step 1: 创建虚拟环境
```
conda create -n lightglue python=3.10
conda activate lightglue
```

## Step 2: 安装 LightGlue-OpenCV  并运行
```
git clone https://github.com/sugarknight999/LightGlue-OpenCV.git
cd LightGlue-OpenCV
python -m pip install -e .
```
## Step3: 运行 demo_camera.py
在终端进入虚拟环境
```
python demo_camera.py
```
即可运行相机并实时获得图像和匹配的特征点

## 效果
结果如图所示
<video src="https://live.csdn.net/v/356455?spm=1001.2014.3001.5501"></video>
https://github.com/sugarknight999/LightGlue-OpenCV/assets/125547075/ef736840-8511-4de9-900f-df4796d7849e


### 项目代码是作者边学习边尝试修改的,代码中含有非常详细的注释,欢迎大家一起学习指正



