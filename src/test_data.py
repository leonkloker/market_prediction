from data import *
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

data = StockData(['SPY'], period='max', data='normalized', features=['Open', 'High', 'Low', 'Close'], additional_features=[])
loader = DataLoader(data, batch_size=1, shuffle=False)
interval = [0.9, 1]
start = int(interval[0] * data.data[0].shape[0])
end = int(interval[1] * data.data[0].shape[0])

for i, (x, y) in enumerate(loader):
    plt.figure()
    plt.plot(np.arange(1, end - start), x[0,start:end,2].numpy())
    plt.savefig('test_data.png')
