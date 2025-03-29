import torch
import inspect

# 映射表：小写的损失函数名称 -> torch.nn模块中的类
criterion_map = {
    'l1loss': torch.nn.L1Loss,
    'mseloss': torch.nn.MSELoss,
    'crossentropyloss': torch.nn.CrossEntropyLoss,
    'ctcloss': torch.nn.CTCLoss,
    'nllloss': torch.nn.NLLLoss,
    'poissonnllloss': torch.nn.PoissonNLLLoss,
    'kldivloss': torch.nn.KLDivLoss,
    'bceloss': torch.nn.BCELoss,
    'bcewithlogitsloss': torch.nn.BCEWithLogitsLoss,
    'marginrankingloss': torch.nn.MarginRankingLoss,
    'hingeembeddingloss': torch.nn.HingeEmbeddingLoss,
    'multilabelmarginaloss': torch.nn.MultiLabelMarginLoss,
    'huberloss': torch.nn.HuberLoss,
    'smoothl1loss': torch.nn.SmoothL1Loss,
    'softmarginloss': torch.nn.SoftMarginLoss,
    'multilabelsoftmarginloss': torch.nn.MultiLabelSoftMarginLoss,
    'cosineembeddingloss': torch.nn.CosineEmbeddingLoss,
    'multimarginloss': torch.nn.MultiMarginLoss,
    'tripletmarginloss': torch.nn.TripletMarginLoss,
    'tripletmarginwithdistanceloss': torch.nn.TripletMarginWithDistanceLoss,
}


def create_criterion(criterion):
    # 如果已经是损失函数实例，直接返回
    if isinstance(criterion, torch.nn.Module):
        return criterion
    # 转换为小写并获取损失函数类
    criterion = criterion['type'].lower()
    criterion_class = criterion_map.get(criterion)
    if not criterion_class:
        raise ValueError(f"Unsupported criterion: {criterion}. "
                         f"Supported options: {list(criterion_map.keys())}")
    # 获取构造函数接受的参数
    signature = inspect.signature(criterion_class.__init__)
    valid_params = list(signature.parameters.keys())[1:]  # 跳过self
    # 过滤无效参数
    filtered_kwargs = {k: v for k, v in criterion['params'].items() if k in valid_params}
    # 实例化损失函数
    return criterion_class(**filtered_kwargs)