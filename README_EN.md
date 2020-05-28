# Age and Gender Estimation

This is a Keras implementation of a CNN for estimating age and gender from a face image [1, 2]. In training, [the IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) is used.  
[**中文**](https://github.com/Acemyzoe/age-gender-estimation/edit/master/README_CN.md)  

## Dependencies

- Python3.7
- Keras or tensorflow.kears
- scipy, numpy, Pandas, tqdm, tables, h5py
- dlib (for demo)
- OpenCV

## Usage

### Create training data from the IMDB-WIKI dataset

First, download the dataset. The dataset is downloaded and extracted to the `data` directory .

Secondly, filter out noise data and serialize images and labels for training into `.mat` file. The training data is created by:

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

### Train network

Train the network using the training data created above.

```bash
python train.py --input data/wiki_crop.mat
```

Trained weight files are stored as `checkpoints/weights.*.hdf5` for each epoch if the validation loss becomes minimum over previous epochs.

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

### Train network with recent data augmentation methods

Recent data augmentation methods, mixup [3] and Random Erasing [4], can be used with standard data augmentation by `--aug` option in training:

```bash
python train.py --input data/wiki_crop.mat --aug
```

Please refer to [this repository](https://github.com/yu4u/mixup-generator) for implementation details.

I confirmed that data augmentation enables us to avoid overfitting and improves validation loss.

### Use the trained network

```
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

Please use the best model among `checkpoints/weights.*.hdf5` for `WEIGHT_FILE` if you use your own trained models.

### Plot training curves from history file

```
python3 plot_history.py --input models/history_16_8.h5 
```

### Network architecture

In [the original paper](https://www.vision.ee.ethz.ch/en/publications/papers/articles/eth_biwi_01299.pdf) [1, 2], the pretrained VGG network is adopted. Here the Wide Residual Network (WideResNet) is trained from scratch. I modified the @asmith26's implementation of the WideResNet; two classification layers (for age and gender estimation) are added on the top of the WideResNet.

Note that while age and gender are independently estimated by different two CNNs in [1, 2], in my implementation, they are simultaneously estimated using a single CNN.

## For further improvement

If you want better results, there would be several options:

- Use larger training images (e.g. --img_size 128).
- Use VGGFace as an initial model and finetune it (https://github.com/rcmalli/keras-vggface).
  - In this case, the size of training images should be (224, 224).
- Use more "clean" dataset (http://chalearnlap.cvc.uab.es/dataset/18/description/) (only for age estimation)

## License

This project is released under the MIT license. However, [the IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) used in this project is originally provided under the following conditions.

> Please notice that this dataset is made available for academic research purpose only. All the images are collected from the Internet, and the copyright belongs to the original owners. If any of the images belongs to you and you would like it removed, please kindly inform us, we will remove it from our dataset immediately.

Therefore, the pretrained model(s) included in this repository is restricted by these conditions (available for academic research purpose only).

## References

[1] R. Rothe, R. Timofte, and L. V. Gool, "DEX: Deep EXpectation of apparent age from a single image," in Proc. of ICCV, 2015.

[2] R. Rothe, R. Timofte, and L. V. Gool, "Deep expectation of real and apparent age from a single image without facial landmarks," in IJCV, 2016.

[3] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond Empirical Risk Minimization," in arXiv:1710.09412, 2017.

[4] Z. Zhong, L. Zheng, G. Kang, S. Li, and Y. Yang, "Random Erasing Data Augmentation," in arXiv:1708.04896, 2017.

[5] E. Agustsson, R. Timofte, S. Escalera, X. Baro, I. Guyon, and R. Rothe, "Apparent and real age estimation in still images with deep residual regressors on APPA-REAL database," in Proc. of FG, 2017.
