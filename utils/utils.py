def normalize(tensor, epsilon=1e-8):
    mean = tensor.mean()
    std = tensor.std(unbiased=False)
    return (tensor - mean) / (std + epsilon)