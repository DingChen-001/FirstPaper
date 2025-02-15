import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed():
    """初始化分布式训练环境"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

class DistributedWrapper:
    """分布式训练封装器（支持多GPU/多节点）"""
    def __init__(self, model, device_ids=None):
        self.local_rank = setup_distributed()
        self.model = DDP(model.to(self.local_rank), 
                        device_ids=[self.local_rank])

    def wrap_data_loader(self, dataset, batch_size):
        """分布式数据加载器"""
        sampler = DistributedSampler(dataset, 
                                   shuffle=True,
                                   num_replicas=dist.get_world_size(),
                                   rank=self.local_rank)
        return DataLoader(dataset, 
                         batch_size=batch_size,
                         sampler=sampler,
                         num_workers=4,
                         pin_memory=True)