from data import *
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

data = StockData(['SPY'], period='max', data='percent', 
                 features=['Volume', 'Open', 'High', 'Low', 'Close'], additional_features=[500],
                 normalization_mask=[False, True, True, True, True], time=True)
loader = DataLoader(data, batch_size=1, shuffle=False)
interval = [0.0, 1]
start = int(interval[0] * data.data[0].shape[0])
end = int(interval[1] * data.data[0].shape[0])

for i, (x, y, t) in enumerate(loader):
    print(x.shape)
    plt.figure()
    plt.plot(np.arange(1, end - start), x[0,start:end,0].numpy())
    plt.savefig('test_data.png')
