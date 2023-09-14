import torch
import sys
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model_dict import DLmodel_dict
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils import (
    get_loaders,
    check_accuracy,
    load_checkpoint,
    save_checkpoint
)

# Hyperparameters etc.
LEARNING_RATE = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 30
NUM_WORKERS = 2
LOAD_MODEL = False
PIN_MEMORY = True
TRAIN_DATA_DIR = '../data/old/train.csv'
VALID_DATA_DIR = '../data/old/valid.csv'

def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for _, (data, label) in enumerate(loop):
        # print(data.shape, label.shape)
        data = data.to(device=DEVICE)
        label = label.float().to(device=DEVICE)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, label)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

# 读取命令行参数
args = sys.argv
num_args = len(args)
modelName = None
if num_args > 1:
    modelName = args[1]

def main():
    if modelName is None:
        print("you need provide a model name")
        assert 0
    best_acc = 0
    model = DLmodel_dict[modelName].to(device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, valid_loader = get_loaders(
        TRAIN_DATA_DIR,
        VALID_DATA_DIR,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY
    )
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler)
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        acc = check_accuracy(valid_loader, model, device=DEVICE)
        if acc > best_acc:
             save_checkpoint(checkpoint, epoch, modelName)
             best_acc = acc
    print(f'model {modelName} get best acc: {best_acc}, the checkpoints is storaged in checkpoints/my_checkpoint')



if __name__ == '__main__':
    main()
