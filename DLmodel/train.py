import torch
import yaml
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

with open('config.yaml', 'r') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)
LEARNING_RATE = float(config['train']['learning_rate'])
mode = config['mode']
BATCH_SIZE = int(config['train']['batch_size'])
NUM_EPOCHS = config['train']['num_epochs']
NUM_WORKERS = config['train']['num_workers']
CKPT_PATH = config['test']['ckpt_path']
PIN_MEMORY = config['pin_memory']
TRAIN_DATA_DIR = config['data']['train_data']
VALID_DATA_DIR = config['data']['valid_data']
modelName = config['model']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def epoch_train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    sum_loss = 0
    for _, (data, label) in enumerate(loop):
        data = data.to(device=DEVICE)
        label = label.float().to(device=DEVICE)
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, label)
            sum_loss += loss
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print(f"the loss is {sum_loss},", end=" ")


def train():
    if modelName is None:
        print("you need provide a model name")
    elif DLmodel_dict.get(modelName) is None:
        print("you need provide a right model name")
    else:
        best_acc, best_epoch, static_epoch, last_acc = 0, -1, 0, 0
        model = DLmodel_dict[modelName].to(device=DEVICE)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train_loader, valid_loader = get_loaders(
            TRAIN_DATA_DIR,
            VALID_DATA_DIR,
            BATCH_SIZE,
            NUM_WORKERS,
            PIN_MEMORY
        )
        scaler = torch.cuda.amp.GradScaler()
        torch.backends.cudnn.enabled = False
        for epoch in range(NUM_EPOCHS):
            print(f"at epoch {epoch},", end=" ")
            epoch_train(train_loader, model, optimizer, loss_fn, scaler)
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            acc = check_accuracy(valid_loader, model, device=DEVICE)
            if acc > best_acc:
                save_checkpoint(checkpoint, epoch, modelName)
                best_epoch = epoch
                best_acc = acc
            if acc == last_acc:
                static_epoch += 1
                if static_epoch == 10:
                    print("the acc has not changed in 10 epochs, so stop the training")
                    print(f'{modelName} get best acc: {best_acc} at epoch {best_epoch}, the checkpoints is storaged in '
              f'checkpoints/my_checkpoint')
                    sys.exit()
            else:
                static_epoch = 0
            last_acc = acc
        print(f'{modelName} get best acc: {best_acc} at epoch {best_epoch}, the checkpoints is storaged in '
              f'checkpoints/my_checkpoint')


def test():
    if modelName is None:
        print("you need provide a model name")
    elif DLmodel_dict.get(modelName) is None:
        print("you need provide a right model name")
    else:
        # TODO
        ...


def main():
    if mode == 'train':
        train()
    else:
        test()


if __name__ == '__main__':
    main()
