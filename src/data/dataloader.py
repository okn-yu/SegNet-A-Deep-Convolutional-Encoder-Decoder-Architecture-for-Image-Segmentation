import torch
from src.data.camvid_dataset import CamVidDataSet as camvid_dataset

# train_dataset = torchvision.datasets.CIFAR10(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
# test_dataset = torchvision.datasets.CIFAR10(root='./data/', train=False, transform=transforms.ToTensor(), download=True)

train_dataset = camvid_dataset(train=True)
test_dataset = camvid_dataset(train=False)

def dataloader():
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8, shuffle=False, num_workers=2)

    return train_loader, test_loader


# train_loader, test_loader = dataloader()
# # for i, (images, labels) in enumerate(train_loader):
# #     print(i, images.shape, labels.shape)