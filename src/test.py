from data import *
from torch.utils.data import DataLoader

from transformer import *
from utils import *

# Model parameters
N_FEATURES = 1
N_EMBEDDING = 32
N_HEADS = 4
N_FORWARD = 32
N_ENC_LAYERS = 1
N_DEC_LAYERS = 1
DEC_WINDOW = 2
ENC_WINDOW = 9
MEM_WINDOW = 2

# Training parameters
NUM_EPOCHS = 300
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
DROPOUT = 0.1

# Data parameters
FEATURES = ['Close']
NORMALIZATION = [True]
ADDITIONAL_FEATURES = []
DATA = 'normalized'
BINARY = 0

STOCK = ['SPY']
VAL_END = 0.9
PERIOD = 'max'

# Log name
date = '2023-07-27-07:09'
stock_str = '_'.join(STOCK)
name = 'transformer_binary{}_{}_features{}_embed{}_enclayers{}_declayers{}_heads{}_foward{}_encwindow{}_decwindow{}_memwindow{}_epochs{}_lr{:.0E}_dropout{}_stocks{}_{}'.format(
        BINARY, DATA, N_FEATURES, N_EMBEDDING, N_ENC_LAYERS, N_DEC_LAYERS, N_HEADS, N_FORWARD, ENC_WINDOW, DEC_WINDOW, MEM_WINDOW, NUM_EPOCHS, LEARNING_RATE, DROPOUT, stock_str, date)

model = Transformer(N_FEATURES, N_EMBEDDING, N_HEADS, N_ENC_LAYERS, N_DEC_LAYERS, N_FORWARD, binary=BINARY, d_pos=N_HEADS)
model.load_state_dict(torch.load('../outputs/models/{}.pth'.format(name)))
model.eval()

test_dataset = StockData(STOCK, PERIOD, data=DATA, binary=BINARY, 
                         features=FEATURES, additional_features=ADDITIONAL_FEATURES, 
                         normalization_mask=NORMALIZATION)
dataloader_test = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

net_values = 0
net_values_baseline = 0
daily_return = 0
daily_volatility = 0
max_drawdown = 0
max_drawdown_baseline = 0
sharpe_ratio = 0
prediction_accuracy = 0
l1_error = 0
l1_error_baseline = 0
avg_prediction = 0
std_prediction = 0

for k, (x, y) in enumerate(dataloader_test):
    start = int(VAL_END * x.shape[-2])
    end = x.shape[-2]
    trading_days = end - start

    prediction = np.squeeze(model(x, x, enc_window=ENC_WINDOW, dec_window=DEC_WINDOW, mem_window=MEM_WINDOW).detach().numpy())
    x = np.squeeze(x.detach().numpy())
    y = np.squeeze(y.detach().numpy())

    avg_prediction = np.mean(prediction[start:end])
    std_prediction = np.std(prediction[start:end])
    trading_signals = trading_strategy(prediction, binary=BINARY, data=DATA)

    prediction_accuracy = accuracy(prediction[start:end], y[start:end], torch=False, data=DATA)

    if DATA == 'normalized':
        net_values = get_net_value(trading_signals[start:end], y[start:end], data=DATA, 
                               mean=test_dataset.mean[0], std=test_dataset.std[0])
        net_values_baseline = get_net_value(np.ones(trading_days), y[start:end], data=DATA,
                               mean=test_dataset.mean[0], std=test_dataset.std[0])

    elif DATA == 'percent':
        net_values = get_net_value(trading_signals[start:end], y[start:end], data=DATA)
        net_values_baseline = get_net_value(np.ones(trading_days), y[start:end], data=DATA)

    if not BINARY:
        l1_error = np.mean(np.abs(prediction[start:end] - y[start:end]))
        l1_error_baseline = np.mean(np.abs(y[start:end]))

    daily_return = np.diff(net_values)
    daily_volatility = np.std(daily_return)

    max_drawdown = get_max_drawdown(net_values)
    max_drawdown_baseline = get_max_drawdown(net_values_baseline)

    sharpe_ratio = (np.mean(daily_return) - net_values_baseline[-1]/trading_days) / daily_volatility

    print("Trading strategy for stock {}:".format(STOCK[k]))
    print("After {} trading days".format(trading_days))
    print("Binary accuracy: {}".format(prediction_accuracy))
    print("Fraction of long signals: {}".format(np.sum(trading_signals[start:end] == 1) / trading_days))
    print("Fraction of short signals: {}".format(np.sum(trading_signals[start:end] == -1) / trading_days))
    print("Overall long return: {}".format(net_values_baseline[-1]))
    print("Overall return: {}".format(net_values[-1]))
    print("Yearly long return: {}".format(net_values_baseline[-1] * 252 / trading_days))
    print("Yearly return: {}".format(net_values[-1] * 252 / trading_days))
    print("Daily volatility: {}".format(daily_volatility))
    print("Max drawdown baseline: {}".format(max_drawdown_baseline))
    print("Max drawdown: {}".format(max_drawdown))
    print("Sharpe ratio: {}".format(sharpe_ratio))
    if not BINARY:
        print("L1 error baseline: {}".format(l1_error_baseline))
        print("L1 error: {}".format(l1_error))
    print("Average prediction: {}".format(avg_prediction))
    print("Std prediction: {}".format(std_prediction))
