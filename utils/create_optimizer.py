import torch
import inspect

optim_map = {
    'adadelta': torch.optim.Adadelta,
    'adafactor': torch.optim.Adafactor,
    'adagrad': torch.optim.Adagrad,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'adamax': torch.optim.Adamax,
    'sparseadam': torch.optim.SparseAdam,
    'asgd': torch.optim.ASGD,
    'lbfgs': torch.optim.LBFGS,
    'nadam': torch.optim.NAdam,
    'rmsprop': torch.optim.RMSprop,
    'rprop': torch.optim.Rprop,
    'sgd': torch.optim.SGD,
}

def create_optimizer(model_params, optimizer_name, **kwargs):
    if isinstance(optimizer_name, torch.optim.Optimizer):
        return optimizer_name
    # 转换为小写并获取优化器类
    optimizer_name = optimizer_name.lower()
    optimizer_class = optim_map.get(optimizer_name)
    if not optimizer_class:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. "
                         f"Supported options: {list(optim_map.keys())}")
    # 获取优化器构造函数接受的参数
    signature = inspect.signature(optimizer_class.__init__)
    valid_params = list(signature.parameters.keys())[1:]  # 跳过self
    # 过滤无效参数
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return optimizer_class(model_params, **filtered_kwargs)