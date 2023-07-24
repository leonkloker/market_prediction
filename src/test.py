from data import *
from torch.utils.data import DataLoader

from transformer import *
from strategy import *
from utils import *

N_FEATURES = 36
N_EMBEDDING = 64
N_HEADS = 8
N_FORWARD = 64
N_ENC_LAYERS = 2
N_DEC_LAYERS = 2
DEC_WINDOW = -1
ENC_WINDOW = -1
MEM_WINDOW = -1
BINARY = 0

NUM_EPOCHS = 200
LEARNING_RATE = 0.001
DROPOUT = 0.1
BATCH_SIZE = 1
STOCKS = ['AAPL']
TEST_INTERVAL = [0.8, 1.0]
PERIOD = 'max'

test_data = MultiStockData(STOCKS, 'max', binary=BINARY)
dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False)

date = '2023-07-24-08:21'
stock_str = '_'.join(STOCKS)
name = 'transformer_binary{}_features{}_embed{}_enclayers{}_declayers{}_heads{}_foward{}_encwindow{}_decwindow{}_memwindow{}_epochs{}_lr{:.0E}_dropout{}_stocks{}_{}'.format(
        BINARY, N_FEATURES, N_EMBEDDING, N_ENC_LAYERS, N_DEC_LAYERS, N_HEADS, N_FORWARD, ENC_WINDOW, DEC_WINDOW, MEM_WINDOW, NUM_EPOCHS, LEARNING_RATE, DROPOUT, stock_str, date)

model = Transformer(N_FEATURES, N_EMBEDDING, N_HEADS, N_ENC_LAYERS, N_DEC_LAYERS, N_FORWARD, binary=BINARY, d_pos=N_HEADS)
model.load_state_dict(torch.load('../outputs/models/{}.pth'.format(name)))
model.eval()

returns = []
daily_volatility = []
max_drawdown = []
sharpe_ratio = []
accuracies = []
l1_losses = []
l1_losses_ref = []
long = []
for k, (x, y) in enumerate(dataloader_test):
    trading_days = int((TEST_INTERVAL[1] - TEST_INTERVAL[0]) * x.shape[-2])
    predictions = (model(x, x, enc_window=ENC_WINDOW, dec_window=DEC_WINDOW, mem_window=MEM_WINDOW).squeeze(0).detach().numpy())
    trading_signals = trading_strategy(predictions, binary=BINARY)
    x = x.squeeze(0).detach().numpy()
    y = y.squeeze(0).detach().numpy()

    accuracies.append(accuracy(predictions[int(x.shape[-2] * TEST_INTERVAL[0]) : int(x.shape[-2] * TEST_INTERVAL[1]), :],
                               y[int(x.shape[-2] * TEST_INTERVAL[0]) : int(x.shape[-2] * TEST_INTERVAL[1]), :], torch=False))
    long.append(np.prod(x[int(TEST_INTERVAL[0] * x.shape[-2]) : int(TEST_INTERVAL[1] * x.shape[-2]), -1] + 1) - 1)

    if not BINARY:
        l1_losses.append(np.mean(np.abs(predictions - x[1:, -1])[int(TEST_INTERVAL[0] * x.shape[-2]) : int(TEST_INTERVAL[1] * x.shape[-2])]))
        l1_losses_ref.append(np.mean(np.abs(x[1:, -1])[int(TEST_INTERVAL[0] * x.shape[-2]) : int(TEST_INTERVAL[1] * x.shape[-2])]))

    daily_returns = (trading_signals[:-1, 0] * x[1:, -1])[int(TEST_INTERVAL[0] * x.shape[-2]) : int(TEST_INTERVAL[1] * x.shape[-2])]
    daily_volatility.append(np.std(daily_returns))

    net_values = np.cumsum(daily_returns)
    i = np.argmax(np.maximum.accumulate(net_values) - net_values)
    if i > 0:
        j = np.argmax(net_values[:i])
    else:
        j = 0 
    max_drawdown.append((net_values[j] - net_values[i]) / (net_values[j] + 1))

    sharpe_ratio.append((np.mean(daily_returns) - long[-1]/trading_days) / daily_volatility[-1])
    returns.append(net_values[-1])

    print("Trading strategy for stock {}:".format(STOCKS[k]))
    print("After {} trading days".format(trading_days))
    print("Binary accuracy: {}".format(accuracies[k]))
    print("Long return: {}".format(long[k]))
    print("Net value: {}".format(np.mean(returns[k])))
    print("Average yearly return: {}".format(np.mean(returns[k]) * 252 / trading_days))
    print("Daily volatility: {}".format(np.mean(daily_volatility[k])))
    print("Max drawdown: {}".format((max_drawdown[k])))
    print("Sharpe ratio: {}".format((sharpe_ratio[k])))
    if not BINARY:
        print("L1 loss: {}".format(l1_losses[k]))
        print("L1 loss reference: {}".format(l1_losses_ref[k]))
    print("")
