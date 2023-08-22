from torch.utils.data import DataLoader

from data import *
from transformer import *
from utils import *

def test_model(N_FEATURES, N_EMBEDDING, N_HEADS, N_ENC_LAYERS, N_DEC_LAYERS, 
               N_FORWARD, ENC_WINDOW, DEC_WINDOW, MEM_WINDOW, NUM_EPOCHS, 
               LEARNING_RATE, DROPOUT, STOCK, FEATURES, NORMALIZATION, ADDITIONAL_FEATURES, 
               DATA, BINARY, TIME, VAL_END, PERIOD, date, file=False):
    """Function to test the performance of a trained transformer model on a given stock
       via several different metrics.

    Args:
        N_FEATURES (int): Amount of features the transformer was trained on.
        N_EMBEDDING (int): Embedding size of the transformer.
        N_HEADS (int): Amount of heads of the transformer.
        N_ENC_LAYERS (int): Amount of encoding blocks of the transformer.
        N_DEC_LAYERS (int): Amount of decoding blocks of the transformer.
        N_FORWARD (int): Size of the feedforward layer of the transformer.
        ENC_WINDOW (int): Attention mask window size of the encoder. (-1 for full causal attention)
        DEC_WINDOW (int): Attention mask window size of the decoder. (-1 for full causal attention)
        MEM_WINDOW (int): Attention mask window size of the cross-attention layer. 
                            (-1 for full causal attention)
        NUM_EPOCHS (int): Amount of epochs the transformer was trained on.
        LEARNING_RATE (float): Initial learning rate during training.
        DROPOUT (float): Dropout rate of the transformer.
        STOCK (List): Contains the ticker symbols of the stocks to be loaded as strings.
        FEATURES (List):  List of all the features that should be loaded from yfinance.
        NORMALIZATION (List): List of bools describing which of the original
                            features should be normalized with mean and std of the 
                            closing prices (True) or with their own mean and std (False).
        ADDITIONAL_FEATURES (List): List of amount of days for which the 
                            moving averages and volatilities of all original features
                            should be calculated and also used as additional features.  
        DATA (string): 'percent' for percent change normalization, 'normalized' for
                            gaussian normalization.
        BINARY (bool): Whether the task should be to predict the next 
                            closing price (False) or only if the price increases
                            or decreases (True).
        TIME (bool): If year, month, date and weekday should be used for the 
                            positional encoding of the transformer.
        VAL_END (float): Fraction where the test data should start relative
                            to all loaded data.
        PERIOD (string): Describes how far back in time the data should be loaded.
        date (string): Date when the transformer training started.
        file (bool, optional): If True, the test summary is appended to the training 
                            log file of the transformer. Defaults to False.
    """
    
    # Name of transformer log and model file
    stock_str = '_'.join(STOCK)
    name = 'transformer_binary{}_{}_features{}_embed{}_enclayers{}_declayers{}_heads{}_foward{}_encw{}_decw{}_memw{}_epochs{}_lr{:.0E}_dropout{}_stocks{}_{}'.format(
            BINARY, DATA, N_FEATURES, N_EMBEDDING, N_ENC_LAYERS, N_DEC_LAYERS, N_HEADS, N_FORWARD, ENC_WINDOW, DEC_WINDOW, MEM_WINDOW, NUM_EPOCHS, LEARNING_RATE, DROPOUT, stock_str, date)

    # Initialize log file
    if file:
        file = open("../outputs/logs/{}.txt".format(name), "a+", encoding="utf-8")
    else:
        file = None

    # Initialize model
    model = Transformer(N_FEATURES, N_EMBEDDING, N_HEADS, N_ENC_LAYERS, N_DEC_LAYERS, N_FORWARD, binary=BINARY, d_pos=N_HEADS, time=TIME)
    model.load_state_dict(torch.load('../outputs/models/{}.pth'.format(name)))
    model.eval()

    # Initialize test data loader
    test_dataset = StockData(STOCK, PERIOD, data=DATA, binary=BINARY, 
                            features=FEATURES, additional_features=ADDITIONAL_FEATURES, 
                            normalization_mask=NORMALIZATION, time=TIME)
    dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Test model
    for k, (x, y, t) in enumerate(dataloader_test):
        
        # start and end of test interval
        start = int(VAL_END * x.shape[-2])
        end = x.shape[-2]
        trading_days = end - start

        # Get prediction
        if TIME:
            prediction = np.squeeze(model(x, x, enc_window=ENC_WINDOW, dec_window=DEC_WINDOW, mem_window=MEM_WINDOW, t=t).detach().numpy())
        else:
            prediction = np.squeeze(model(x, x, enc_window=ENC_WINDOW, dec_window=DEC_WINDOW, mem_window=MEM_WINDOW).detach().numpy())

        x = np.squeeze(x.detach().numpy())
        y = np.squeeze(y.detach().numpy())

        # Calculate average and standard deviation of predictions
        avg_prediction = np.mean(prediction[start:end])
        std_prediction = np.std(prediction[start:end])

        # Calculate trading signals and binary accuracy
        trading_signals = trading_strategy(prediction, y, binary=BINARY, data=DATA)
        prediction_accuracy = accuracy(prediction[start:end], y[start:end], torch=False, data=DATA, binary=BINARY)
        
        if BINARY:
            y = x[1:, -1]
            trading_signals = trading_signals[:-1]
            trading_days -= 1

        # Calculate net values throughout the test interval
        if DATA == 'normalized':
            net_values = get_net_value(trading_signals[start:end], y[start:end], data=DATA, 
                                mean=test_dataset.mean[0], std=test_dataset.std[0])
            net_values_baseline = get_net_value(np.ones_like(y[start:end]), y[start:end], data=DATA,
                                mean=test_dataset.mean[0], std=test_dataset.std[0])

        elif DATA == 'percent':
            net_values = get_net_value(trading_signals[start:end], y[start:end], data=DATA)
            net_values_baseline = get_net_value(np.ones_like(y[start:end]), y[start:end], data=DATA)

        # Calculate L1 error if the model is not binary
        if not BINARY:
            l1_error = np.mean(np.abs(prediction[start:end] - y[start:end]))
            l1_error_baseline = np.mean(np.abs(y[start:end-1] - y[start+1:end]))

        # Calculate daily volatility, max drawdown and sharpe ratio
        daily_return = np.diff(net_values)
        daily_volatility = np.std(daily_return)

        max_drawdown = get_max_drawdown(net_values)
        max_drawdown_baseline = get_max_drawdown(net_values_baseline)

        sharpe_ratio = (np.mean(daily_return) - net_values_baseline[-1]/trading_days) / daily_volatility

        # Print test summary
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
    N_EMBEDDING = 64
    N_HEADS = 16
    N_FORWARD = 64
    N_ENC_LAYERS = 1
    N_DEC_LAYERS = 3
    DEC_WINDOW = 10
    ENC_WINDOW = -1
    MEM_WINDOW = 10

    # Training parameters
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 1
    DROPOUT = 0.2
    WARMUP = 0

    # Data parameters
    FEATURES = ['Volume', 'Open', 'High', 'Low', 'Close']
    NORMALIZATION = [False, True, True, True, True]
    ADDITIONAL_FEATURES = [5, 10, 50, 100, 500]
    DATA = 'percent'
    BINARY = 1
    TIME_FEATURES = 1

    STOCK = ['ADDDF']
    VAL_END = 0.8
    PERIOD = 'max'

    N_FEATURES = len(FEATURES) * (len(ADDITIONAL_FEATURES) * 2 + 1)

    # Log date
    date = '2023-08-21-08:14'

    test_model(N_FEATURES, N_EMBEDDING, N_HEADS, N_ENC_LAYERS, N_DEC_LAYERS, 
                N_FORWARD, ENC_WINDOW, DEC_WINDOW, MEM_WINDOW, NUM_EPOCHS, 
                LEARNING_RATE, DROPOUT, STOCK, FEATURES, NORMALIZATION, ADDITIONAL_FEATURES, 
                DATA, BINARY, TIME_FEATURES, VAL_END, PERIOD, date=date)