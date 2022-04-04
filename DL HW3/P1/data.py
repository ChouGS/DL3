import torch
import random

def get_data():
    # data_list = []
    # label_list = []
    data_dim = random.randint(10, 100)
    data = [random.randint(0, 999) for _ in range(data_dim)]
    label = [max(data)]
    #     data_list.append(torch.Tensor(data).unsqueeze(0))
    #     label_list.append(torch.Tensor(label).unsqueeze(0))
    # data = torch.cat(data_list, 0)
    # label = torch.cat(label_list, 0)
    # delimiter = int(round(data_list.shape[0] * test_frac))

    return torch.Tensor(data).unsqueeze(1), torch.Tensor(label).unsqueeze(1)

