import numpy as np
import torch
from matplotlib import pyplot as plot
import os

from model import SetNet
from data import get_data

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    model = SetNet()
    training_num = 900
    testing_num = 100

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = torch.nn.MSELoss()
    loss_log = []

    for e in range(5):
        print(f'============= EPOCH {e} =============')
        model.train()
        # training
        for i in range(training_num):
            iter_data, iter_label = get_data()
            iter_data = iter_data.unsqueeze(0)
            iter_label = iter_label.unsqueeze(0)
            logit = model(iter_data)
            loss_val = loss(logit, iter_label)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            loss_log.append(loss_val.item())
            if i % 50 == 0:
                print(f'Training {i}/{training_num}: Loss={round(loss_val.item(), 6)}')

        model.eval()
        test_eval_items = []
        # testing
        for i in range(testing_num):
            iter_data, iter_label = get_data()
            iter_data = iter_data.unsqueeze(0)
            iter_label = iter_label.unsqueeze(0)
            logit = model(iter_data)
            test_eval_items.append(loss(logit, iter_label))
        
        avg_eval_loss = sum(test_eval_items) / len(test_eval_items)

        print(f'\nTesting: avg loss={avg_eval_loss}\n')

    loss_log = np.log10(np.array(loss_log))
    index = np.arange(loss_log.shape[0])

    plot.plot(index, loss_log, 'b-')
    plot.ylabel('Logged Loss')
    plot.xlabel('Iter')
    plot.title('Training Loss Curve for P1 (Logged)')
    plot.savefig('Loss.png')
