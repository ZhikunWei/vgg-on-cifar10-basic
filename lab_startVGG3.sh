#!/bin/bash
/home/wzk/anaconda3/bin/python main_v3.py --log-number v3_0 --batch-size 256 --dist-url 'tcp://cluster1:20452' --pretrained --multiprocessing-distributed --world-size 3 --rank 2 dataset/cifar10