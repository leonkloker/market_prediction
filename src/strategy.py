import numpy as np

def trading_strategy(predictions, binary=False, mode='percent'):
    if binary:
        signal = (predictions > 0.5).astype(int)
        signal[signal == 0] = -1

    elif mode == 'percent':
        signal = np.zeros_like(predictions)
        signal[predictions > 0] = 1
        signal[predictions < 0] = -1

    elif mode == 'normalized':
        signal = np.convolve(predictions, np.array([1, -1]), mode='same')
        signal[signal > 0] = 1
        signal[signal < 0] = -1
    
    return signal
