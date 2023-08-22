import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader

from data import *

# Which stock and features to look at
data = StockData(['SPY'], period='max', data='normalized', 
                 features=['Volume', 'Open', 'High', 'Low', 'Close'], additional_features=[100, 500],
                 normalization_mask=[False, True, True, True, True], time=True, binary=False)

loader = DataLoader(data, batch_size=1, shuffle=False)

# interval of the entire timeseries to be plotted
interval = [0.5, 1]
start = int(interval[0] * data.data[0].shape[0])
end = int(interval[1] * data.data[0].shape[0])

feature_idx = np.array([-1, -11, -21])

for i, (x, y, t) in enumerate(loader):
    print(y.shape)
    plt.figure()
    plt.plot(np.arange(1, end - start), x[0, start:end, feature_idx].numpy())
    plt.xlabel('Time / days')
    plt.ylabel('Feature')
    plt.savefig('test_data.png')
