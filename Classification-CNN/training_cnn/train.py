import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from .model import CNN

"""
Training and evaluation script for CNN to classify an image of size 128x128x3 RGB.
"""
def main():
    train_data = torchvision.datasets.ImageFolder(root='data/train/train', transform=torchvision.transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    model = CNN()
    model.cuda()
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            #Check accuracy as well
            acc = (output.argmax(dim=1) == target).float().mean()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.3f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), acc.item()))
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    main() 
