import torch
import torch.nn as nn
from data import MultiStockData
from transformer import Transformer

STOCK = ['AAPL']

N_FEATURES = 4
N_EMBEDDING = 64
N_HEADS = 4
N_FORWARD = 64
N_ENC_LAYERS = 0
N_DEC_LAYERS = 1
DEC_WINDOW = -1
ENC_WINDOW = -1
MEM_WINDOW = -1

NUM_EPOCHS = 300
LEARNING_RATE = 0.001

date = '2023-07-14-05:53'
name = 'transformer_features{}_embed{}_enclayers{}_declayers{}_heads{}_foward{}_encwindow{}_decwindow{}_memwindow{}_epochs{}_lr{:.0E}_{}'.format(
        N_FEATURES, N_EMBEDDING, N_ENC_LAYERS, N_DEC_LAYERS, N_HEADS, N_FORWARD, ENC_WINDOW, DEC_WINDOW, MEM_WINDOW, NUM_EPOCHS, LEARNING_RATE, date)

test_data = MultiStockData(STOCK, 'max', use_interval=[0, 1])
dataloader_test = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

model = Transformer(N_FEATURES, N_EMBEDDING, N_HEADS, N_ENC_LAYERS, N_DEC_LAYERS, N_FORWARD)
model.load_state_dict(torch.load('../outputs/models/{}.pth'.format(name)))
model.eval()

l1loss = nn.L1Loss()

l1_model = 0
l1_baseline = 0
binary_model = 0
binary_baseline = 0

for x, y in dataloader_test:
    eval_start = int(0.9 * x.size(-2))
    pred = model(x, x, enc_window=ENC_WINDOW, dec_window=DEC_WINDOW, mem_window=MEM_WINDOW)
    l1_model += l1loss(pred[:, eval_start:, :], y[:, eval_start:, :])
    l1_baseline += l1loss(torch.zeros((1, x.size(-2) - eval_start, 1)), y[:, eval_start:, :])
    binary_model += torch.mean((torch.sign(pred[:, eval_start:, :]) == torch.sign(y[:, eval_start:, :])).float())
    binary_baseline += torch.mean((1 == torch.sign(y[:, eval_start:, :])).float())

l1_model /= len(dataloader_test)
l1_baseline /= len(dataloader_test)
binary_model /= len(dataloader_test)
binary_baseline /= len(dataloader_test)

print("L1 loss of model: {}".format(l1_model))
print("L1 loss of baseline: {}".format(l1_baseline))
print("Binary accuracy of model: {}".format(binary_model))
print("Binary accuracy of bullish baseline: {}".format(binary_baseline))
