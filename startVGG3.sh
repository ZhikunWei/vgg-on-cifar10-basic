#!/bin/bash
sudo /home/a2019211150/anaconda3/bin/python main_v2.py --dist-url 'tcp://nasp-cpu-01-v4:20452' --pretrained  --multiprocessing-distributed --world-size 3 --rank 2 dataset/cifar10