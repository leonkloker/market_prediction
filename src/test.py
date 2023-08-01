from torch.utils.data import DataLoader

from data import *
from transformer import *
from utils import *

def test_model(N_FEATURES, N_EMBEDDING, N_HEADS, N_ENC_LAYERS, N_DEC_LAYERS, 
               N_FORWARD, ENC_WINDOW, DEC_WINDOW, MEM_WINDOW, NUM_EPOCHS, 
               LEARNING_RATE, DROPOUT, STOCK, FEATURES, NORMALIZATION, ADDITIONAL_FEATURES, 
               DATA, BINARY, TIME, VAL_END, PERIOD, date, file=False):
    
    stock_str = '_'.join(STOCK)
    name = 'transformer_binary{}_{}_features{}_embed{}_enclayers{}_declayers{}_heads{}_foward{}_encwindow{}_decwindow{}_memwindow{}_epochs{}_lr{:.0E}_dropout{}_stocks{}_{}'.format(
            BINARY, DATA, N_FEATURES, N_EMBEDDING, N_ENC_LAYERS, N_DEC_LAYERS, N_HEADS, N_FORWARD, ENC_WINDOW, DEC_WINDOW, MEM_WINDOW, NUM_EPOCHS, LEARNING_RATE, DROPOUT, stock_str, date)

    if file:
        file = open("../outputs/logs/{}.txt".format(name), "a+", encoding="utf-8")
    else:
        file = None

    model = Transformer(N_FEATURES, N_EMBEDDING, N_HEADS, N_ENC_LAYERS, N_DEC_LAYERS, N_FORWARD, binary=BINARY, d_pos=N_HEADS, time=TIME)
    model.load_state_dict(torch.load('../outputs/models/{}.pth'.format(name)))
    model.eval()

    test_dataset = StockData(STOCK, PERIOD, data=DATA, binary=BINARY, 
                            features=FEATURES, additional_features=ADDITIONAL_FEATURES, 
                            normalization_mask=NORMALIZATION, time=TIME)
    dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=True)

    for k, (x, y, t) in enumerate(dataloader_test):
        start = int(VAL_END * x.shape[-2])
        end = x.shape[-2]
        trading_days = end - start

        if TIME:
            prediction = np.squeeze(model(x, x, enc_window=ENC_WINDOW, dec_window=DEC_WINDOW, mem_window=MEM_WINDOW, t=t).detach().numpy())
        else:
            prediction = np.squeeze(model(x, x, enc_window=ENC_WINDOW, dec_window=DEC_WINDOW, mem_window=MEM_WINDOW).detach().numpy())

        x = np.squeeze(x.detach().numpy())
        y = np.squeeze(y.detach().numpy())

        avg_prediction = np.mean(prediction[start:end])
        std_prediction = np.std(prediction[start:end])

        trading_signals = trading_strategy(prediction, binary=BINARY, data=DATA)

        prediction_accuracy = accuracy(prediction[start:end], y[start:end], torch=False, data=DATA, binary=BINARY)

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

        print("Trading strategy for stock {}:".format(STOCK[k]), file=file)
        print("After {} trading days".format(trading_days), file=file)
        print("Binary accuracy: {:.5f}".format(prediction_accuracy), file=file)
        print("Fraction of long signals: {:.5f}".format(np.sum(trading_signals[start:end] == 1) / trading_days), file=file)
        print("Fraction of short signals: {:.5f}".format(np.sum(trading_signals[start:end] == -1) / trading_days), file=file)
        print("Overall long return: {:.5f}".format(net_values_baseline[-1]), file=file)
        print("Overall return: {:.5f}".format(net_values[-1]), file=file)
        print("Yearly long return: {:.5f}".format(net_values_baseline[-1] * 252 / trading_days), file=file)
        print("Yearly return: {:.5f}".format(net_values[-1] * 252 / trading_days), file=file)
        print("Daily volatility: {:.5f}".format(daily_volatility), file=file)
        print("Max drawdown baseline: {:.5f}".format(max_drawdown_baseline), file=file)
        print("Max drawdown: {:.5f}".format(max_drawdown), file=file)
        print("Sharpe ratio: {:.5f}".format(sharpe_ratio), file=file)
        if not BINARY:
            print("L1 error baseline: {:.5f}".format(l1_error_baseline), file=file)
            print("L1 error: {:.5f}".format(l1_error), file=file)
        print("Average prediction: {:.5f}".format(avg_prediction), file=file)
        print("Std prediction: {:.5f}".format(std_prediction), file=file)
        print("\n", file=file)

    if file:
        file.close()

if __name__ == '__main__':

    # Model parameters
    N_FEATURES = 11
    N_EMBEDDING = 32
    N_HEADS = 4
    N_FORWARD = 32
    N_ENC_LAYERS = 3
    N_DEC_LAYERS = 3
    DEC_WINDOW = 10
    ENC_WINDOW = 10
    MEM_WINDOW = 10

    # Training parameters
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-3
    DROPOUT = 0.2

    # Data parameters
    FEATURES = ['Close']
    NORMALIZATION = [True]
    ADDITIONAL_FEATURES = [5, 10, 20, 50, 100]
    DATA = 'normalized'
    BINARY = 0

    STOCK = ['SPY']
    VAL_END = 0.95
    PERIOD = 'max'

    # Log date
    date = '2023-07-28-09:02'

    test_model(N_FEATURES, N_EMBEDDING, N_HEADS, N_ENC_LAYERS, N_DEC_LAYERS, 
                N_FORWARD, ENC_WINDOW, DEC_WINDOW, MEM_WINDOW, NUM_EPOCHS, 
                LEARNING_RATE, DROPOUT, STOCK, FEATURES, NORMALIZATION, ADDITIONAL_FEATURES, 
                DATA, BINARY, VAL_END, PERIOD, date)