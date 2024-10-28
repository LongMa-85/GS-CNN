import csv

from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import *
from torchvision.utils import save_image

from net001 import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
weight_path = 'params_xiugai4/unet_256_0.pth'
data_path = 'train_img'
save_path = 'train_xiugai4_zhongjian_img'
log_path = 'training_log_xiugai4.csv'

# 创建日志文件并写入表头
if not os.path.exists(log_path):
    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss'])

def log_training(epoch,  train_loss):
    with open(log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss])
if __name__ == '__main__':
    data_loader = DataLoader(MyDataset(data_path), batch_size=32, shuffle=True)
    net = UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight!')
    else:
        print("not successful load weight!")

    opt = optim.Adam(net.parameters(),lr=0.0001)
    loss_fun = nn.MSELoss()
    epoch = 1
    while True:
        total_train_loss = 0.0
        num_batches = 0
        for i, (HG, GS) in enumerate(data_loader):
            HG, GS = HG.to(device).to(torch.float32), GS.to(device).to(torch.float32)
            out_image = net(HG)
            train_loss = loss_fun(out_image, GS)
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            total_train_loss += train_loss.item()
            num_batches += 1

            if (i % 1) == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')
                _HG = HG[0]
                _GS = GS[0]
                _out = out_image[0]
                img = torch.stack([_GS, _HG, _out], dim=0)
                save_image(img, f'{save_path}/{i // 1}.png')

            if (epoch % 10) == 0:
                torch.save(net.state_dict(), f'params_xiugai4/unet_256_{epoch}.pth')

        # 计算平均损失
        avg_train_loss = total_train_loss / num_batches

        # 每轮记录一次日志
        if (epoch % 1) == 0:
            log_training(epoch, avg_train_loss)

        epoch += 1