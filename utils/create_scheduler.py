import torch
import inspect

# 映射表，将调度器名称映射到对应的PyTorch类
scheduler_map = {
    'lambda': torch.optim.lr_scheduler.LambdaLR,
    'step': torch.optim.lr_scheduler.StepLR,
    'multistep': torch.optim.lr_scheduler.MultiStepLR,
    'exponential': torch.optim.lr_scheduler.ExponentialLR,
    'cosineannealing': torch.optim.lr_scheduler.CosineAnnealingLR,
    'reduceonplateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'cyclic': torch.optim.lr_scheduler.CyclicLR,
    'onecycle': torch.optim.lr_scheduler.OneCycleLR,
    'cosineannealingwarmrestarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
}


def create_scheduler(optimizer, scheduler):
    # 如果scheduler已经是实例，直接返回
    if isinstance(scheduler, (torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
        return scheduler

    # 转换为小写并获取调度器类
    scheduler_name = scheduler['type'].lower()
    scheduler_class = scheduler_map.get(scheduler_name)
    if not scheduler_class:
        supported = list(scheduler_map.keys())
        raise ValueError(f"Unsupported scheduler: {scheduler_name}. Supported options: {supported}")

    # 获取构造函数参数列表
    signature = inspect.signature(scheduler_class.__init__)
    valid_params = list(signature.parameters.keys())[1:]  # 跳过self参数

    # 过滤无效参数
    filtered_kwargs = {k: v for k, v in scheduler['params'].items() if k in valid_params}

    # 实例化调度器
    return scheduler_class(optimizer, **filtered_kwargs)