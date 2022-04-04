import torch
from torch.utils.data import Dataset
import sklearn.datasets as dataset

class SwissRollDataset(Dataset):
    def __init__(self, N) -> None:
        super(SwissRollDataset, self).__init__()
        data, _ = dataset.make_swiss_roll(N)
        self.data = torch.Tensor(data)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data[index]
