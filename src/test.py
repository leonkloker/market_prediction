from data import *
from torch.utils.data import DataLoader

stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'TSLA']
train_data = MultiStockData(stocks, 'max', use_interval=[0, 0.8])
val_data = MultiStockData(stocks, 'max', use_interval=[0.8, 0.9])
test_data = MultiStockData(stocks, 'max', use_interval=[0.9, 1])

dataloader_train = DataLoader(train_data, batch_size=1, shuffle=True)
dataloader_val = DataLoader(val_data, batch_size=1, shuffle=True)
dataloader_test = DataLoader(test_data, batch_size=1, shuffle=True)

for x, y in dataloader_train:
    print(x[0,1,:], y[0,0,:])
    break
