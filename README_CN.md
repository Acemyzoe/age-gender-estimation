# 年龄和性别估计

这是CNN的Keras实现，用于根据面部图像估算年龄和性别[1、2]。在训练中，使用[了IMDB-WIKI数据集](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)。

## 依赖

- Python3.7
- Keras或tensorflow.kears
- scipy，numpy，Pandas，tqdm，表格，h5py
- dlib（用于演示）
- OpenCV的

## 用法

### 从IMDB-WIKI数据集中创建训练数据

首先，下载数据集。数据集已下载并提取到`data`目录中。

其次，过滤掉噪声数据并序列化图像和标签以训练成`.mat`文件。培训数据是通过以下方式创建的：

```bash
python create_db.py --output data/wiki_db.mat --db wiki --img_size 64
usage: create_db.py [-h] --output OUTPUT [--db DB] [--img_size IMG_SIZE] [--min_score MIN_SCORE]

This script cleans-up noisy labels and creates database for training.

optional arguments:
  -h, --help                 show this help message and exit
  --output OUTPUT, -o OUTPUT path to output database mat file (default: None)
  --db DB                    dataset; wiki or imdb (default: wiki)
  --img_size IMG_SIZE        output image size (default: 32)
  --min_score MIN_SCORE      minimum face_score (default: 1.0)
```

### 火车网络

使用上面创建的训练数据训练网络。

```bash
python train.py --input data/wiki_crop.mat
```

`checkpoints/weights.*.hdf5`如果验证损失比以前的时期最小，则按每个时期存储训练的权重文件。

```
usage: train.py [-h] --input INPUT [--batch_size BATCH_SIZE]
                [--nb_epochs NB_EPOCHS] [--lr LR] [--opt OPT] [--depth DEPTH]
                [--width WIDTH] [--validation_split VALIDATION_SPLIT] [--aug]
                [--output_path OUTPUT_PATH]

This script trains the CNN model for age and gender estimation.

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        path to input database mat file (default: None)
  --batch_size BATCH_SIZE
                        batch size (default: 32)
  --nb_epochs NB_EPOCHS
                        number of epochs (default: 30)
  --lr LR               initial learning rate (default: 0.1)
  --opt OPT             optimizer name; 'sgd' or 'adam' (default: sgd)
  --depth DEPTH         depth of network (should be 10, 16, 22, 28, ...)
                        (default: 16)
  --width WIDTH         width of network (default: 8)
  --validation_split VALIDATION_SPLIT
                        validation split ratio (default: 0.1)
  --aug                 use data augmentation if set true (default: False)
  --output_path OUTPUT_PATH
                        checkpoint dir (default: checkpoints)
```

### 使用最新的数据扩充方法训练网络

最近的数据增强方法，混合[3]和随机擦除[4]，可以通过`--aug`在培训中选择使用标准数据增强：

```bash
python train.py --input data/wiki_crop.mat --aug
```

请参阅[此存储库](https://github.com/yu4u/mixup-generator)以获取实现详细信息。

我确认数据增强使我们能够避免过度拟合并改善验证损失。

### 使用预训练网络

```bash
python demo.py
usage: demo.py [-h] [--weight_file WEIGHT_FILE] [--depth DEPTH]
               [--width WIDTH] [--margin MARGIN] [--image_dir IMAGE_DIR]

This script detects faces from web cam input, and estimates age and gender for
the detected faces.

optional arguments:
  -h, --help            show this help message and exit
  --weight_file WEIGHT_FILE
                        path to weight file (e.g. weights.28-3.73.hdf5)
                        (default: None)
  --depth DEPTH         depth of network (default: 16)
  --width WIDTH         width of network (default: 8)
  --margin MARGIN       margin around detected face for age-gender estimation (default: 0.4)
  --image_dir IMAGE_DIR
                        target image directory; if set, images in image_dir
                        are used instead of webcam (default: None)
```

### 网络架构

在[原始论文](https://www.vision.ee.ethz.ch/en/publications/papers/articles/eth_biwi_01299.pdf) [1、2]中，采用了预训练的VGG网络。在这里，广泛的残余网络（WideResNet）从头开始进行培训。我修改了@ asmith26的WideResNet的实现；WideResNet的顶部添加了两个分类层（用于年龄和性别估计）。

请注意，虽然年龄和性别是由[1、2]中两个不同的CNN独立估算的，但在我的实现中，它们是同时使用单个CNN估算的。

## 进一步改进

如果您想要更好的结果，将有几种选择：

- 使用较大的训练图像（例如--img_size 128）。

- 使用VGGFace作为初始模型并对其进行微调（

  https://github.com/rcmalli/keras-vggface

  ）。

  - 在这种情况下，训练图像的大小应为（224，224）。

- 使用更多的“干净”数据集（http://chalearnlap.cvc.uab.es/dataset/18/description/）（仅用于年龄估计）

## 执照

该项目是根据MIT许可发布的。但是，本项目中使用的[IMDB-WIKI数据集](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)最初是在以下条件下提供的。

> 请注意，此数据集仅可用于学术研究目的。所有图像均从互联网上收集，版权归原始所有者所有。如果有任何图像属于您，并且您希望将其删除，请告知我们，我们会立即将其从数据集中删除。

因此，此存储库中包含的预训练模型受这些条件的限制（仅可用于学术研究目的）。

## 参考文献

[1] R. Rothe, R. Timofte, and L. V. Gool, "DEX: Deep EXpectation of apparent age from a single image," in Proc. of ICCV, 2015.

[2] R. Rothe, R. Timofte, and L. V. Gool, "Deep expectation of real and apparent age from a single image without facial landmarks," in IJCV, 2016.

[3] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond Empirical Risk Minimization," in arXiv:1710.09412, 2017.

[4] Z. Zhong, L. Zheng, G. Kang, S. Li, and Y. Yang, "Random Erasing Data Augmentation," in arXiv:1708.04896, 2017.

[5] E. Agustsson, R. Timofte, S. Escalera, X. Baro, I. Guyon, and R. Rothe, "Apparent and real age estimation in still images with deep residual regressors on APPA-REAL database," in Proc. of FG, 2017.
