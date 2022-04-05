import torch
import random

def get_data():
    data_dim = random.randint(10, 100)
    data = [random.randint(0, 999) for _ in range(data_dim)]
    label = [max(data)]

    return torch.Tensor(data), torch.Tensor(label)

