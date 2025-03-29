import numpy as np
import torch
def sample_to_tensor(data, device = None):
    converted = []
    for idx, item in enumerate(data):
        if idx == 1: # Action!
            tensor = torch.tensor(np.array(item), dtype=torch.long)
        else:
            tensor = torch.tensor(np.array(item), dtype=torch.float)
        if device is not None:
            tensor = tensor.to(device)
        converted.append(tensor)
    return tuple(converted)