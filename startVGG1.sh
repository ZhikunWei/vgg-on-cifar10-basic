#!/bin/bash
sudo /home/a2019211150/anaconda3/bin/python main.py -a vgg19 --dist-url 'tcp://\[2001:da8:bf:15:6ccd:cdb9:3c69:fcbd\]:22' --pretrained --dist-backend 'gloo' --multiprocessing-distributed --world-size 3 --rank 0 dataset/cifar10