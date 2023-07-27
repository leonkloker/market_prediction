from data import *
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

data = StockData(['SPY'], period='max', data='percent', features=['Open', 'High', 'Low', 'Close'], additional_features=[])
loader = DataLoader(data, batch_size=1, shuffle=False)

for i, (x, y) in enumerate(loader):
    plt.figure()
    plt.plot(np.arange(0, x.shape[1]), x[0,:,2].numpy())
    plt.savefig('test_data_{}.png'.format(i))
