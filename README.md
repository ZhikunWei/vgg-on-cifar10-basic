# Distributed vgg on cifar10 (Future Internet Project)
Contact me if there is a problem running the codes.
#final result
1.Number of Epochs: 30

2.Training/Testing Accuracy achieved: 87.3% / 71.7%

3.Training/Testing accuracy w.r.t epochs: (figure) 'plot_results/figures/acc_train_test.png'

4.Total training time: 451.8 minutes

5.Average time per epoch: 14.9 minutes

6.Actual time per epoch(including training and testing): (figure) 'plot_results/figures/training_time.png'

7.Communication cost per epoch, failed to take note of every epoch, so on average, one epoch cost: (unit:GiB)

                rx      tx
    server1     54.8   54.8
    server2     54.8    54.6
    server3     54.4    54.8

8.Total communication cost duraing entire training process(30 epochs):(unit: GiB)

                rx      tx
    server1     1664.2    1664.1
    server2     1664.3    1638.6
    server3     1632.4    1664.7


###To run the codes
1. download the files into server or your computer.<br>
    >git  clone https://github.com/ZhikunWei/vgg-on-cifar10-basic.git
2. update files into the servers.<br>
    > cd vgg-on-cifar10-basic <br>
    chmod 777 updatefiles.sh <br>
    ./updatefiles.sh <br>
    <input passwords "2019211150" three times(for three servers)>

3. To run the codes:

(on server1):
>cd vgg1/ <br>
>./startVGG1.sh
    
(on server2):
>cd vgg2/ <br>
>./startVGG2.sh

(on server3):
>cd vgg3/ <br>
>./startVGG3.sh

4.using vnstat to monitor the traffic
>vnstat -l
        



----------------------------------------------------------------------------------------------------
# working log

## before
Find torch.distributed module and learn about it

## 12.19
main_v2.py: 
Finish baseline model on 3 nodes: parameter server pattern, synchronous every batch

#####improvement:
adjust batch size on each node, so that all nodes complete the 
#### torch.distributed
Parameters are never broadcast between processes. The module performs
        an all-reduce step on gradients and assumes that they will be modified
        by the optimizer in all processes in the same way. Buffers
        (e.g. BatchNorm stats) are broadcast from the module in process of rank
        0, to all other replicas in the system in every iteration.

b128 = 8G
b200 = 19-5=14
b256 = 22-5 = 17

baseline result: one epoch [131 batch]
####server1:
bytes: rx 343.41 GiB  | tx 338.58 GiB <br>
Acc@1 85.060 Acc@5 99.550

####server2
bytes: rx 343.11 GiB | tx 344.63 GiB <br>
Acc@1 85.060 Acc@5 99.550

####server3  
bytes rx 339.16 GiB  | tx 343.02 GiB <br>
Acc@1  85.060 Acc@5 99.550

            b32     b64     b128
    server1 11.5    21.5    42
    server2 45    
    server3 6-7.3   12      60-90
    
fraze 3 layers  
batch0  RX bytes:3643928721143  TX bytes:3665941853884  
batch5  RX bytes:3647953571364  TX bytes:3669990808732  62.5,57,67
batch20 RX bytes:3658015499936 (3.6 TB)  TX bytes:3680114323840 (3.6 TB)
batch35 RX bytes:3668349515616 (3.6 TB)  TX bytes:3690503657237 (3.6 TB)

froze 2 layers   Acc@1 81.404 Acc@5 99.550
batch0  RX bytes:4053699553625 (4.0 TB)  TX bytes:4080387523933 (4.0 TB)
batch13 RX bytes:4054926379290 (4.0 TB)  TX bytes:4081621849801 (4.0 TB)
batch20 RX bytes:4055586951171 (4.0 TB)  TX bytes:4082286471732 (4.0 TB)
batch123 RX bytes:4065306719780 (4.0 TB)  TX bytes:4092062381075 (4.0 TB)

froze 1 layer  
0  RX bytes:4065972342337 (4.0 TB)  TX bytes:4093897886483 (4.0 TB)
10 RX bytes:4065974735095 (4.0 TB)  TX bytes:4093900231441 (4.0 TB)
131*10 RX bytes:4066297062980 (4.0 TB)  TX bytes:4094203980751 (4.0 TB)

no fraze layers
batch0  RX bytes:3671436998297 (3.6 TB)  TX bytes:3694775601177 (3.6 TB)
batch5  RX bytes:3676135574915 (3.6 TB)  TX bytes:3699502061397 (3.6 TB)
batch131*2+93 RX bytes:4052730933873 (4.0 TB)  TX bytes:4078247662911 (4.0 TB)

  
##1229
one epoch:
 with init: bytes     47.37 GiB  |       48.27 GiB
 without init: bytes      46.80 GiB  |       46.63 GiB
 

