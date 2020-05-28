# age-gender-estimation 年龄性别分析
learn from yu4u : Keras implementation of a CNN network for age and gender estimation.  

[**更详细的自述**](https://github.com/Acemyzoe/age-gender-estimation/blob/master/README_EN.md)  
**BY** [GJ](https://github.com/Acemyzoe/age-gender-estimation.git)

## 使用指南 
1. 运行demo.py使用预训练的模型进行摄像头演示，或者使用--image_dir [IMAGE_DIR]选项在目录中识别图像，预训练的模型保存在pretrained_models文件夹。 
2. 从IMDB-WIKI数据集中创建训练数据，[下载地址](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)  
    使用create_db.py将图像和标签序列化成mat文件。 
   > python create_db.py --output data/wiki_crop.mat --db wiki --img_size 64
3. 使用上面的训练数据训练网络 
     > python train.py --input data/wiki_crop.mat  
   模型保存至checkpoints文件夹（如果验证损失比之前小，则按每个时期存储训练权重文件。 
   可选参数可以用--help显示或者参考源码。
4. 绘制历史训练曲线 
   > python plot_history.py --input models/history_16_8.h5
5. [抛砖引玉](https://github.com/yu4u/age-gender-estimation.git)
## 环境配置
  * ubuntu18.04+python3.7
  * 推荐使用Anaconda（一个提供包管理和环境管理的python版本）。  [官网下载](https://www.anaconda.com/distribution/)
  * 推荐修改镜像地址：
  
      >pip install pip -U 
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  
* 安装需要的python库：(缺少相应的库可用conda或者pip自行安装) 
    > * opencv-python
    > * matplotlib
    > * keras
    > * tensorflow
    > * dlib
