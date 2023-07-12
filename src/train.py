import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.tensorboard import SummaryWriter

from transformer import *
from data import *
from utils import *

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
BATCH_SIZE = 1
STOCKS = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'TSLA']
TRAIN_INTERVAL = [0, 0.8]
VAL_INTERVAL = [0, 0.9]
TEST_INTERVAL = [0, 1]
PERIOD = 'max'

date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
name = 'transformer_features{}_embed{}_enclayers{}_declayers{}_heads{}_foward{}_encwindow{}_decwindow{}_memwindow{}_epochs{}_lr{:.0E}_{}'.format(
        N_FEATURES, N_EMBEDDING, N_ENC_LAYERS, N_DEC_LAYERS, N_HEADS, N_FORWARD, ENC_WINDOW, DEC_WINDOW, MEM_WINDOW, NUM_EPOCHS, LEARNING_RATE, date)

file = open("../outputs/logs/{}.txt".format(name), "w", encoding="utf-8")
writer = SummaryWriter(log_dir="../outputs/tensorboards/{}".format(name))

dataloader = DataLoader(MultiStockData(STOCKS, PERIOD, TRAIN_INTERVAL), batch_size=BATCH_SIZE, shuffle=True)
dataloader_val = DataLoader(MultiStockData(STOCKS, PERIOD, VAL_INTERVAL), batch_size=BATCH_SIZE, shuffle=True)
dataloader_test = DataLoader(MultiStockData(STOCKS, PERIOD, TEST_INTERVAL), batch_size=BATCH_SIZE, shuffle=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using cuda", file=file)
else:
    device = torch.device('cpu')
    print("Using CPU", file=file)

model = Transformer(N_FEATURES, N_EMBEDDING, N_HEADS, N_ENC_LAYERS, N_DEC_LAYERS, N_FORWARD)
model = model.to(device)
model_info = summary(model, input_size=[(BATCH_SIZE, 200, N_FEATURES), (BATCH_SIZE, 200, N_FEATURES)])
print(model_info, file=file)
file.flush()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=10)
criterion = nn.MSELoss()
val_loss_min = np.Inf

for epoch in range(NUM_EPOCHS):
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        outputs = model(x, x, enc_window=ENC_WINDOW, dec_window=DEC_WINDOW, mem_window=MEM_WINDOW)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in dataloader_val:
            x = x.to(device)
            y = y.to(device)
            out = model(x, x, enc_window=ENC_WINDOW, dec_window=DEC_WINDOW, mem_window=MEM_WINDOW)
            val_loss += criterion(out[:,int(x.size(-2)*TRAIN_INTERVAL[1]):,:], y[:,int(x.size(-2)*TRAIN_INTERVAL[1]),:])
    val_loss = val_loss / len(dataloader_val)

    scheduler.step(val_loss)

    writer.add_scalar("MSE/train", loss, epoch)
    writer.add_scalar("MSE/val", val_loss, epoch)

    if val_loss < val_loss_min:
        torch.save(model.state_dict(), 
                   '../outputs/models/{}.pth'.format(name))
        val_loss_min = val_loss

    model.train()

    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, train MSE: {loss.item()}", file=file)
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, val MSE: {val_loss}", file=file)
    file.flush()


print("Training finished", file=file)
print("\nEvaluating best model on test set", file=file)
file.flush()

model.load_state_dict(torch.load('../outputs/models/{}.pth'.format(name)))

model.eval()
test_loss = 0
with torch.no_grad():
    for x, y in dataloader_test:
        x = x.to(device)
        y = y.to(device)
        out = model(x, x, enc_window=ENC_WINDOW, dec_window=DEC_WINDOW, mem_window=MEM_WINDOW)
        test_loss += criterion(out[:,int(x.size(-2)*VAL_INTERVAL[1]):,:], y[:,int(x.size(-2)*VAL_INTERVAL[1]),:])
test_loss = test_loss / len(dataloader_test)

print(f"test loss: {test_loss}", file=file)

writer.add_scalar("MSE/test", test_loss, epoch)
file.flush()
file.close()
writer.close()
