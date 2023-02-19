# AgiProbot_Motor_Segmentation

This project introduces a way to solve industrial automation problems using deep learning. Industrial problems : the factory has collected a lot of discarded motors, but there are a lot of parts in them that can be reused, so we need to disassemble the discarded motors, so we need to get the exact location of the screws. Our solution is to use deep learning to semantically segment and utilize the 3D point cloud collected by the sensor. The result of the division gives the position of the screw

# Environments Requirement

CUDA = 10.2

Python = 3.7.0

PyTorch = 1.6

Open3d

tpdm

API(kmeans_pytorch) 

For pytorch, you could directly install all the dependence according to the instruction on pytorch official website. After you install the evironment successfully, you could use pip to install open3d and tpdm. For the kmeans_pytorch, you could follow the instruction(https://pypi.org/project/kmeans-pytorch/)

# Three different train entries

train_semseg is the training entry for 10 categories semantic segmantation and types(typeA typeB) classification

train_semseg_and_two_classification is the training entry for 10 catefories semantic segmentation and other two classifications(Type and existence of cover)(with one opt)

train_semseg_and_two_classification_two_opt is the training entry for 10 catefories semantic segmentation and other two classifications(Type and existence of cover)(with two opts:first opt to update the tail producing the classification result and second opt(using the result of classification from first opt) producing the segmentaion result)

# How to run

## Training the pretraining model

You can use below command line to run the pretraining script and gain the pretraining model:
```
CUDA_VISIBLE_DEVICES=0,1 python train_semseg.py --batch_size 16 --npoints 2048 --epoch 100 --model PCT --lr 0.01   --exp_name STN_16_2048_100 --factor_stn_los 0.01 --kernel_loss_weight 0.05 --use_class_weight 0 --screw_weight 1 --which_dataset Dataset4 --num_segmentation_type 10 --emb_dims 1024 --train 1 --finetune 0 --test 0 --data_dir /home/ies/bi/data/Dataset4
```

| cmd  | Description          | Type | Property |
| ------- | ----------------------------------------------------------| --- | ---------- |
| -batch_size | batch size for training process                |      int |   obligatory      |
| -npoints   | number of points for sub point cloud               |     int  |      obligatory      |
| -epoch   |  training epoch                               | int      | obligatory |
| -model   | the model to be chosed                                 | string     | obligatory |
| -lr | initial learning rate                                 | string     | obligatory |
| -exp_name   | experimential name which include some parameters of current trained model  | string    | obligatory   |
| -factor_stn_los  | the weigth of loss for STN Network  | float | obligatory |
| -kernel_loss_weight   | the weigth of loss for patch sample Network   | float |  obligatory  |
| -use_class_weight   | whether to use the class weight    | int | obligatory |
| -screw_weight   | the bolts weights  | float | optional |
| -which_dataset | the current dataset you use to train the model | string  | obligaroty |
| -num_segmentation_type | the number of categories you want to classify | int | obligatory |
| -emb_dims   | the dimension for high features setting     | int  | obligatory  |
| -train   | if we are in the training process    | int | obligatory |
| -finetune   | if we are in the finetune process | int | obligatory |
| -data_dir   | the position where the dataset is stored | string | obligatory |

## Train the finetune model
You can use below command line to run the finetune script and gain the training result:
```
CUDA_VISIBLE_DEVICES=0,1 python segmentation_train_test.py --batch_size 16 --npoints 2048 --epoch 100 --model dgcnn --lr 0.01   --exp_name STN_16_2048_100 --factor_stn_los 0.01 --kernel_loss_weight 0.05 --use_class_weight 0 --screw_weight 1 --which_dataset Dataset4 --num_segmentation_type 6 --emb_dims --train 0 --finetune 1 --test 0 --data_dir /home/ies/bi/data/finetune
```
When we want to finetune on a specific pretrained model, we should set the exp_name exactly same with taht in the pretraining process, and we set the initial learning rate as 0.001, we should also set the train as 0 and finetune as 1. the data_dir also should be set as the position where the real-world dataset stored.

## Test the finetune model
You can use below command line to run the test finetune script and gain the test result:
```
CUDA_VISIBLE_DEVICES=0 python segmentation_train_test.py --batch_size 16 --npoints 2048 --epoch 100 --model PCT --lr 0.01   --exp_name STN_16_2048_100 --factor_stn_los 0.01 --kernel_loss_weight 0.05 --use_class_weight 0 --screw_weight 1 --which_dataset Dataset4 --num_segmentation_type 10 --emb_dims 1024 --train 0 --finetune 1 --test 1 --data_dir /home/ies/bi/data/testdata
```

Before you run this command line, you should firstly put the finetuned model(.pth file) under the directory trained_model(in the name of best_finetune and best_finetune_patch respectively)
