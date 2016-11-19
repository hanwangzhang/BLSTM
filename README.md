# Play and Rewind: Optimizing Binary Representations of Videos by Self-Supervised Temporal Hashing (oral)
Hanwang Zhang, Meng Wang, Richang Hong, Tat-Seng Chua.
ACM MM 2016 


## Citation
```
@InProceedings{lu2016visual,
   title = {Play and Rewind: Optimizing Binary Representations of Videos by Self-Supervised Temporal Hashing},
   author = {Zhang, Hanwang and Wang, Meng and Chua, Tat-Seng},
   booktitle = {MM},
   year = {2016},
 }
 ```

## Introduction
An unsupervised hashing model that generates binary codes (+1,-1) for a video sequence. This is just a quick demo for running the training and test. The source code is simple and well commented. Future details about feature extraction and visualization will be added ASAP.


## Enviroment Requirements
Only [Theano] (http://deeplearning.net/software/theano/) is required. In fact, some of the core layers exploit a high-level wrapper [Keras] (https://keras.io/), but the code is not dependent on Keras installation.
You may need to install **h5py** for data loader.

## Demo
```
>> run Blstm.py;
```
