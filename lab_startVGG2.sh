#!/bin/bash
sudo /home/wzk/anaconda3/bin/python main_v2.py --dist-url 'tcp://cluster1:20452' --pretrained --multiprocessing-distributed --world-size 3 --rank 1 dataset/cifar10