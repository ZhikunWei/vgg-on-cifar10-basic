# vgg-on-cifar10-basic

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

baseline result: 
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


no fraze layers
batch0  RX bytes:3671436998297 (3.6 TB)  TX bytes:3694775601177 (3.6 TB)
batch5  RX bytes:3676135574915 (3.6 TB)  TX bytes:3699502061397 (3.6 TB)


  

