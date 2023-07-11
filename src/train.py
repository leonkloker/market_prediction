import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from data import MultiStockData
from transformer import Transformer

# Dataset parameters
STOCKS = ['AAPL'] * 1
STOCKS += ['MSFT', 'GOOG', 'AMZN', 'FB', 'TSLA', 'NVDA', 'PYPL', 'ADBE', 'NFLX']
PERIOD = 'max'
TRAIN_INTERVAL = [0, 0.8]
VAL_INTERVAL = [0.8, 0.9]
TEST_INTERVAL = [0.9, 1]

# Model parameters
D_FEATURES = 5
D_MODEL = 256
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
D_FEEDFORWARD = 256
WINDOW = 20
DROPOUT = 0.1

# Training parameters
EPOCHS = 100
BATCH_SIZE = 1
LR = 0.001

# Log name
name = "transformer_features{}_embed{}_enclayers{}_declayers{}_heads{}_forward{}_dropout{}_window{}_epochs{}_lr{:.0E}_{}".format(
    D_FEATURES, D_MODEL, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, NUM_HEADS, D_FEEDFORWARD, DROPOUT, 20, EPOCHS, LR, PERIOD)
file = open("../outputs/logs/{}.txt".format(name), "w", encoding="utf-8")

# Dataloader definition
train_dataset = MultiStockData(STOCKS, PERIOD, TRAIN_INTERVAL)
val_dataset = MultiStockData(STOCKS, PERIOD, VAL_INTERVAL)
test_dataset = MultiStockData(STOCKS, PERIOD, TEST_INTERVAL)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using cuda", file=file)
else:
    device = torch.device('cpu')
    print("Using CPU", file=file)

# Model definition
model = Transformer(D_FEATURES, D_MODEL, NUM_HEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, 
                    D_FEEDFORWARD, dropout=DROPOUT, window=WINDOW)
model = model.to(device)
model_info = summary(model, input_size=[(BATCH_SIZE, 200, D_FEATURES), 
                                 (BATCH_SIZE, 200, D_FEATURES)])
print(model_info, file=file)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
val_loss_min = float('inf')

# Training loop
for epoch in range(EPOCHS):
    for x, y in train_dataloader:
        
        x = x.to(device)
        y = y.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(x, x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_dataloader:
            x = x.to(device)
            y = y.to(device)
            out = model(x, x)
            val_loss += criterion(out, y)
    val_loss = val_loss / len(val_dataloader)

    if val_loss < val_loss_min:
        torch.save(model.state_dict(), 
                '../outputs/models/{}.pth'.format(name))
        val_loss_min = val_loss

    model.train()

    print(f"Epoch {epoch+1} / {EPOCHS}, train loss: {loss.item()}", file=file)
    print(f"Epoch {epoch+1} / {EPOCHS}, val loss: {val_loss}", file=file)
    file.flush()

file.close()
