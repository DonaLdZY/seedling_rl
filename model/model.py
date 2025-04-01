import torch
from utils.create_optimizer import create_optimizer
from utils.create_scheduler import create_scheduler
import os

class Model:
    def __init__(self, module, **kwargs):
        self.device = kwargs.get('device', torch.device('cpu'))
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        self.module = module.to(self.device)
        try:
            self.optimizer = create_optimizer(self.module.parameters(), kwargs['optimizer'])
        except KeyError:
            raise KeyError('optimizer is required')
        self.scheduler = create_scheduler(self.optimizer, kwargs['scheduler']) if 'scheduler' in kwargs else None

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def backward(self, loss):
        loss.backward()

    def step(self):
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()

    def save_checkpoint(self, save_dir, tag=None, client_state=None):
        if client_state is None:
            client_state = {}
        os.makedirs(save_dir, exist_ok=True)

        # 构造状态字典
        checkpoint = {
            'model': self.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'client_state': client_state
        }

        # 生成文件名
        filename = f"checkpoint_{tag}.pth" if tag else "checkpoint.pth"
        save_path = os.path.join(save_dir, filename)

        # 保存文件
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, load_dir, tag=None, load_module_only=False):
        # 生成文件路径
        filename = f"checkpoint_{tag}.pth" if tag else "checkpoint.pth"
        load_path = os.path.join(load_dir, filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No checkpoint found at {load_path}")

        # 加载状态字典
        checkpoint = torch.load(load_path, map_location=self.device)
        # 加载模型参数
        self.module.load_state_dict(checkpoint['model'])
        # 选择性加载其他状态
        if not load_module_only:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.scheduler and checkpoint['scheduler'] is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler'])

        print(f"Checkpoint loaded from {load_path}")
        return checkpoint.get('client_state', {})

