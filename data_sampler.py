from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import math
import torch


class MyDistributedSampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        The sampler is adjusted to sample different partition to different
        node according to their batch size

    Arguments:
        dataset: Dataset used for sampling.
        partition: the batch sizes of different nodes, in our case, is of three nodes
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, partition=[87, 23, 96], num_replicas=None, rank=None):
        super(DistributedSampler, self).__init__(dataset)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.partition = partition
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.ratio0 = partition[0] * 1.0 / sum(partition)
        self.ratio1 = partition[1] * 1.0 / sum(partition)
        self.ratio2 = partition[2] * 1.0 / sum(partition)
        self.num_samples_node0 = int(math.ceil(len(self.dataset) * self.ratio0))
        self.num_samples_node1 = int(math.ceil(len(self.dataset) * self.ratio1))
        self.num_samples_node2 = int(math.ceil(len(self.dataset) * self.ratio2))
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples_node0 + self.num_samples_node1 + self.num_samples_node2

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        if self.rank == 0:
            indices = indices[0:self.num_samples_node0]
            print('number of image assigned to node0:', len(indices))
        elif self.rank == 1:
            indices = indices[self.num_samples_node0: self.num_samples_node0+self.num_samples_node1]
            print('number of image assigned to node1', len(indices))
        elif self.rank == 2:
            indices = indices[self.num_samples_node0+self.num_samples_node1: self.total_size]
            print('number of image assigned to node2', len(indices))

        return iter(indices)

    def __len__(self):
        if self.rank == 0:
            return self.num_samples_node0
        elif self.rank == 1:
            return self.num_samples_node1
        elif self.rank == 2:
            return self.num_samples_node2

    def set_epoch(self, epoch):
        self.epoch = epoch
