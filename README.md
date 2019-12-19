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

  

