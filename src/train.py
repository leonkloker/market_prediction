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

# Model parameters
N_FEATURES = 36
N_EMBEDDING = 64
N_HEADS = 8
N_FORWARD = 64
N_ENC_LAYERS = 4
N_DEC_LAYERS = 4
DEC_WINDOW = -1
ENC_WINDOW = -1
MEM_WINDOW = -1

# Training parameters
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
DROPOUT = 0.1
WARMUP = 50

# Data parameters
STOCKS_WARMUP = ['SPY', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA', 'META', 'TSLA']
STOCKS_FINETUNE = ['ETH-USD']
TRAIN_END = 0.9
VAL_END = 0.95
PERIOD = 'max'
BINARY = 0

# Log name
date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
stock_str = '_'.join(STOCKS_FINETUNE)
name = 'transformer_binary{}_features{}_embed{}_enclayers{}_declayers{}_heads{}_foward{}_encwindow{}_decwindow{}_memwindow{}_epochs{}_lr{:.0E}_dropout{}_stocks{}_{}'.format(
        BINARY, N_FEATURES, N_EMBEDDING, N_ENC_LAYERS, N_DEC_LAYERS, N_HEADS, N_FORWARD, ENC_WINDOW, DEC_WINDOW, MEM_WINDOW, NUM_EPOCHS, LEARNING_RATE, DROPOUT, stock_str, date)

# Create log file and tensorboard writer
file = open("../outputs/logs/{}.txt".format(name), "w", encoding="utf-8")
writer = SummaryWriter(log_dir="../outputs/tensorboards/{}".format(name))

# Create dataloaders
dataloader_warmup = DataLoader(MultiStockData(STOCKS_WARMUP, PERIOD, end=TRAIN_END, binary=BINARY), batch_size=BATCH_SIZE, shuffle=True)
dataloader = DataLoader(MultiStockData(STOCKS_FINETUNE, PERIOD, end=TRAIN_END, binary=BINARY), batch_size=BATCH_SIZE, shuffle=True)
dataloader_val = DataLoader(MultiStockData(STOCKS_FINETUNE, PERIOD, end=VAL_END, binary=BINARY), batch_size=BATCH_SIZE, shuffle=True)
dataloader_test = DataLoader(MultiStockData(STOCKS_FINETUNE, PERIOD, end=1, binary=BINARY), batch_size=BATCH_SIZE, shuffle=True)

# Check if cuda is available
if torch.cuda.is_available():
    device = torch.device('cuda:7')
    print("Using cuda", file=file)
else:
    device = torch.device('cpu')
    print("Using CPU", file=file)

# Create model, optimizer and scheduler
model = Transformer(N_FEATURES, N_EMBEDDING, N_HEADS, N_ENC_LAYERS, N_DEC_LAYERS, N_FORWARD, binary=BINARY, dropout=DROPOUT, d_pos=N_HEADS)
model = model.to(device)
model_info = summary(model, input_size=[(BATCH_SIZE, 200, N_FEATURES), (BATCH_SIZE, 200, N_FEATURES)])
print(model_info, file=file)
file.flush()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

# Define loss function
if BINARY:
    criterion = nn.BCELoss()
else:
    criterion = nn.MSELoss()

val_loss_min = np.Inf

# Train model
for epoch in range(NUM_EPOCHS):
    if epoch < WARMUP:
        for x, y in dataloader_warmup:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x, x, enc_window=ENC_WINDOW, dec_window=DEC_WINDOW, mem_window=MEM_WINDOW)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    else:
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x, x, enc_window=ENC_WINDOW, dec_window=DEC_WINDOW, mem_window=MEM_WINDOW)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    # Evaluate model on validation set
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for x, y in dataloader_val:
            x = x.to(device)
            y = y.to(device)
            out = model(x, x, enc_window=ENC_WINDOW, dec_window=DEC_WINDOW, mem_window=MEM_WINDOW)
            val_loss += criterion(out[:,int(x.size(-2)*TRAIN_END):int(x.size(-2)*VAL_END),:], y[:,int(x.size(-2)*TRAIN_END):int(x.size(-2)*VAL_END),:])
            val_acc += accuracy(out[:,int(x.size(-2)*TRAIN_END):int(x.size(-2)*VAL_END),:], y[:,int(x.size(-2)*TRAIN_END):int(x.size(-2)*VAL_END),:])
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
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, train loss: {loss.item()}", file=file)
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, val loss: {val_loss}", file=file)
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, val acc: {val_acc}", file=file)
    file.flush()

print("Training finished", file=file)
print("\nEvaluating best model on test set", file=file)
file.flush()

# Evaluate best model on test set
model.load_state_dict(torch.load('../outputs/models/{}.pth'.format(name)))
model.eval()
test_loss = 0
test_acc = 0
with torch.no_grad():
    for x, y in dataloader_test:
        x = x.to(device)
        y = y.to(device)
        out = model(x, x, enc_window=ENC_WINDOW, dec_window=DEC_WINDOW, mem_window=MEM_WINDOW)
        test_loss += criterion(out[:,int(x.size(-2)*VAL_END):,:], y[:,int(x.size(-2)*VAL_END):,:])
        test_acc += accuracy(out[:,int(x.size(-2)*VAL_END):,:], y[:,int(x.size(-2)*VAL_END):,:])
test_loss = test_loss / len(dataloader_test)
test_acc = test_acc / len(dataloader_test)

print(f"test loss: {test_loss}", file=file)
print(f"test acc: {test_acc}", file=file)
writer.add_scalar("Loss/test", test_loss, epoch)
writer.add_scalar("Accuracy/test", test_acc, epoch)

file.flush()
file.close()
writer.close()
