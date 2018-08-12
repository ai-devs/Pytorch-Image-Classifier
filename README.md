# Pytorch Image Classifier
This is an image classifier done with __Pytorch__, it makes use of *Densenet* as the pretrained model.

There is a script version which can be used to classify images, it has been created to classify *102 different species of flowers* but can be adapted easily to classify other types of images as well by pre-training the classifier and changing its output layer size for the desired number of categories.

The script also allows to choose resnet as a 2nd architecture, but at the moment resnet is still having some issues when giving out the probabilities and will be addresed in the future, so for the moment please use only densenet.

The **flowers dataset** is available to *download* here:

https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz

## How to use the scripts

### TRAIN

```python train.py flowers --gpu --epochs 5 --learning_rate 0.001 --arch densenet```

```python train.py flowers --gpu --epochs 10 --learning_rate 0.001 --arch densenet --save_dir checkpoints```


### PREDICT

```python predict.py flowers/valid/1/image_06739.jpg```
