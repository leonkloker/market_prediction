import numpy as np

def trading_strategy(predictions, binary=False):
    if binary:
        signal = (predictions > 0.5).astype(int)
        signal[signal == 0] = -1

    else:
        signal = np.zeros_like(predictions)
        signal[predictions > 0.0] = 1
        signal[predictions < -0.0] = -1
    
    return signal