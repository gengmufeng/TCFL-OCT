# TCFL_Unpaired_OCT_Image_Denoising
Pytorch implementation of "Triplet Cross-Fusion Learning for Unpaired Image Denoising in Optical Coherence Tomography."

## File Description

### 1. code
DataLoader_train.py: python script to build pytorch Dataset and DataLoader for training.

DataLoader_test.py: python script to build pytorch Dataset and DataLoader for test.

model.py: the implementation of our proposed TCFL-DnCNN achitectures.

train.py: a basic template python file for training the model.

test.py: a basic template python file for testing the model.

### 2. data
train-clean: clean OCT images for training.

train-noisy: noisy OCT images for training.

test-noisy: noisy OCT images for test.

dataset_division: contain five .txt files, storing the file names of the images in the training set or test set, respectively.

### 3. result
saved_models: save the trained models.

test: save the predicted clean OCT images when testing.

### 4. citation
If you use this code or PKU37 dataset ( https://wiki.milab.wiki/display/LF/Open+Source+Project ) for your research, please cite our paper.

@ARTICLE{9800979,

author={Geng, Mufeng and Meng, Xiangxi and Zhu, Lei and Jiang, Zhe and Gao, Mengdi and Huang, Zhiyu and Qiu, Bin and Hu, Yicheng and Zhang, Yibao and Ren, Qiushi and Lu, Yanye},

journal={IEEE Transactions on Medical Imaging},

title={Triplet Cross-Fusion Learning for Unpaired Image Denoising in Optical Coherence Tomography},

year={2022},

volume={},

number={},

pages={1-1},

doi={10.1109/TMI.2022.3184529}}

