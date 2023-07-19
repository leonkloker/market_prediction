import numpy as np

def trading_strategy(predictions, binary=False):
    if binary:
        signal = (predictions > 0.5).astype(int)
        signal[signal == 0] = -1

    else:
        signal = (predictions > 0).astype(int)
        signal[signal == 0] = -1
    
    return signal