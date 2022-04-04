from collections import OrderedDict
import torch
import torch.nn as nn

class SetNet(nn.Module):
    def __init__(self, num_layers=5) -> None:
        super(SetNet, self).__init__()
        self.model = torch.nn.Sequential(
            OrderedDict([(f'mp{i}', MessagePassing()) for i in range(num_layers)] + \
                        [('pool', nn.AdaptiveMaxPool1d(1))]))
    
    def forward(self, x):
        for layer in self.model:
            if isinstance(layer, nn.AdaptiveMaxPool1d):
                x = torch.transpose(x, 1, 0)
            x = layer(x)
        
        return x

class MessagePassing(nn.Module):
    def __init__(self, hidden_length=32) -> None:
        super(MessagePassing, self).__init__()
        self.message = nn.Linear(1, hidden_length)
        self.readout = nn.Linear(hidden_length, 1)

    def forward(self, x):
        '''
        x: bs x 1
        '''
        x_message = self.message(x) # bs x h
        x = torch.max(x_message, 0, keepdim=True).values
        x = self.readout(x)
        return x
