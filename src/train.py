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
from test import *

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
MIN_INP_SIZE = 100

STOCKS_WARMUP = ['SPY', 'AAPL', 'AMZN', 'GOOGL', 'NVDA', 'META', 'TSLA']
STOCKS_FINETUNE = ['ADDDF']
TRAIN_END = 0.9
VAL_END = 0.95
PERIOD = '3000d'

N_FEATURES = len(FEATURES) * (len(ADDITIONAL_FEATURES) * 2 + 1)
N_PADDING = max(max(N_ENC_LAYERS * (ENC_WINDOW - 1) + MEM_WINDOW - 1, DEC_WINDOW - 1,
                    N_ENC_LAYERS * (ENC_WINDOW - 1) + MEM_WINDOW - 1 + (DEC_WINDOW - 1) * (N_DEC_LAYERS - 1)),
                MIN_INP_SIZE)

# Log name
date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
stock_str = '_'.join(STOCKS_FINETUNE)
name = 'transformer_binary{}_{}_features{}_embed{}_enclayers{}_declayers{}_heads{}_foward{}_encw{}_decw{}_memw{}_epochs{}_lr{:.0E}_dropout{}_stocks{}_{}'.format(
        BINARY, DATA, N_FEATURES, N_EMBEDDING, N_ENC_LAYERS, N_DEC_LAYERS, N_HEADS, N_FORWARD, ENC_WINDOW, DEC_WINDOW, MEM_WINDOW, NUM_EPOCHS, LEARNING_RATE, DROPOUT, stock_str, date)

# Create log file and tensorboard writer
file = open("../outputs/logs/{}.txt".format(name), "w", encoding="utf-8")
writer = SummaryWriter(log_dir="../outputs/tensorboards/{}".format(name))

# Create dataloaders
dataloader_warmup = DataLoader(StockData(STOCKS_WARMUP, PERIOD, end=TRAIN_END, 
                                         data=DATA, binary=BINARY, features=FEATURES,
                                         additional_features=ADDITIONAL_FEATURES, 
                                         normalization_mask=NORMALIZATION, time=TIME_FEATURES), batch_size=BATCH_SIZE, shuffle=True)
dataloader = DataLoader(StockData(STOCKS_FINETUNE, PERIOD, end=TRAIN_END, 
                                         data=DATA, binary=BINARY, features=FEATURES,
                                         additional_features=ADDITIONAL_FEATURES, 
                                         normalization_mask=NORMALIZATION, time=TIME_FEATURES), batch_size=BATCH_SIZE, shuffle=True)
dataloader_val = DataLoader(StockData(STOCKS_FINETUNE, PERIOD, end=1, 
                                         data=DATA, binary=BINARY, features=FEATURES,
                                         additional_features=ADDITIONAL_FEATURES, 
                                         normalization_mask=NORMALIZATION, time=TIME_FEATURES), batch_size=BATCH_SIZE, shuffle=True)
dataloader_test = DataLoader(StockData(STOCKS_FINETUNE, PERIOD, end=1, 
                                         data=DATA, binary=BINARY, features=FEATURES,
                                         additional_features=ADDITIONAL_FEATURES, 
                                         normalization_mask=NORMALIZATION, time=TIME_FEATURES), batch_size=BATCH_SIZE, shuffle=True)

# Check if cuda is available
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("Using cuda", file=file)
else:
    device = torch.device('cpu')
    print("Using CPU", file=file)

# Create model, optimizer and scheduler
model = Transformer(N_FEATURES, N_EMBEDDING, N_HEADS, N_ENC_LAYERS, N_DEC_LAYERS, N_FORWARD, 
                    binary=BINARY, dropout=DROPOUT, d_pos=N_HEADS, time=TIME_FEATURES)
model = model.to(device)
model_info = summary(model, input_size=[(BATCH_SIZE, 5000, N_FEATURES), (BATCH_SIZE, 5000, N_FEATURES)])
print(model_info, file=file)
file.flush()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=10, min_lr=1e-7)

# Define loss function
if BINARY:
    criterion = nn.BCELoss()
else:
    criterion = nn.MSELoss()

val_loss_min = np.Inf

# Train model
for epoch in range(NUM_EPOCHS):
    if epoch < WARMUP:
        for x, y, t in dataloader_warmup:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            if TIME_FEATURES:
                t = t.to(device)
                out = model(x, x, enc_window=ENC_WINDOW, dec_window=DEC_WINDOW, mem_window=MEM_WINDOW, t=t)
            else:
                out = model(x, x, enc_window=ENC_WINDOW, dec_window=DEC_WINDOW, mem_window=MEM_WINDOW)

            loss = criterion(out[:, N_PADDING:, :], y[:, N_PADDING:, :])
            loss.backward()
            optimizer.step()

    else:
        for x, y, t in dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            if TIME_FEATURES:
                t = t.to(device)
                out = model(x, x, enc_window=ENC_WINDOW, dec_window=DEC_WINDOW, mem_window=MEM_WINDOW, t=t)
            else:
                out = model(x, x, enc_window=ENC_WINDOW, dec_window=DEC_WINDOW, mem_window=MEM_WINDOW)

            loss = criterion(out[:, N_PADDING:, :], y[:, N_PADDING:, :])
            loss.backward()
            optimizer.step()

    # Evaluate model on validation set
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for x, y, t in dataloader_val:
            val_start = int(x.size(-2)*TRAIN_END)
            val_end = int(x.size(-2)*VAL_END)
            x = x.to(device)
            y = y.to(device)
            
            if TIME_FEATURES:
                t = t.to(device)
                out = model(x, x, enc_window=ENC_WINDOW, dec_window=DEC_WINDOW, mem_window=MEM_WINDOW, t=t)
            else:
                out = model(x, x, enc_window=ENC_WINDOW, dec_window=DEC_WINDOW, mem_window=MEM_WINDOW)

            val_loss += criterion(out[:, val_start:val_end, :], y[:, val_start:val_end, :])
            val_acc += accuracy(out[:, val_start:val_end, :], y[:, val_start:val_end, :], torch=True, data=DATA, binary=BINARY)
    val_loss = val_loss / len(dataloader_val)
    val_acc = val_acc / len(dataloader_val)
    writer.add_scalar("Loss/train", loss.item(), epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)

    # Save model if validation loss has decreased
    if val_loss < val_loss_min:
        torch.save(model.state_dict(), 
                   '../outputs/models/{}.pth'.format(name))
        val_loss_min = val_loss

    # Update learning rate
    model.train()
    scheduler.step(val_loss)

    # Print epoch results
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, learning rate: {optimizer.param_groups[0]['lr']}", file=file)
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, train loss: {loss.item()}", file=file)
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, val loss: {val_loss}", file=file)
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, val acc: {val_acc}", file=file)
    file.flush()

print("Training finished", file=file)
print("\nEvaluating best model", file=file)
file.flush()
file.close()
writer.close()

# Evaluate best model on entire dataset
test_model(N_FEATURES, N_EMBEDDING, N_HEADS, N_ENC_LAYERS, N_DEC_LAYERS, 
                N_FORWARD, ENC_WINDOW, DEC_WINDOW, MEM_WINDOW, NUM_EPOCHS, 
                LEARNING_RATE, DROPOUT, STOCKS_FINETUNE, FEATURES, NORMALIZATION, ADDITIONAL_FEATURES, 
                DATA, BINARY, TIME_FEATURES, 0, PERIOD, date, file=True)

# Evaluate best model on test dataset
test_model(N_FEATURES, N_EMBEDDING, N_HEADS, N_ENC_LAYERS, N_DEC_LAYERS,
            N_FORWARD, ENC_WINDOW, DEC_WINDOW, MEM_WINDOW, NUM_EPOCHS, 
            LEARNING_RATE, DROPOUT, STOCKS_FINETUNE, FEATURES, NORMALIZATION, ADDITIONAL_FEATURES, 
            DATA, BINARY, TIME_FEATURES, VAL_END, PERIOD, date, file=True)
