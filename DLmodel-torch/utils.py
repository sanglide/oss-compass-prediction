import torch
import os
from dataset import OssDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, epoch, name):
    print("=====> Saving checkpoint <=====")
    path = "checkpoints/" + name + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + "checkpoint_best"+ ".pth.tar"
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=====> Loading checkpoint <=====")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
        train_data_dir,
        valid_daya_dir,
        batch_size,
        num_workers=4,
        pin_memory=True,
):
    train_ds = OssDataset(train_data_dir)
    train_loader = DataLoader(train_ds, 
                              batch_size=batch_size, 
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True)
    valid_ds = OssDataset(valid_daya_dir)
    valid_loader = DataLoader(valid_ds,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=False)
    return train_loader, valid_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            # 计算每行的最大值
            max_values = torch.max(preds, dim=1)[0]
            # 创建一个形状与原始张量相同的全零张量
            result_tensor = torch.zeros_like(preds)
            # 使用 torch.where 将每行的最大值设置为1，其他值设置为0
            for i in range(preds.shape[0]):
                result_tensor[i] = torch.where(preds[i] == max_values[i], 1, 0)
                if result_tensor[i].equal(y[i].to(torch.float32)):
                    num_correct += 1
                num_pixels += 1
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels * 100:.2f}%")
    return num_correct/num_pixels