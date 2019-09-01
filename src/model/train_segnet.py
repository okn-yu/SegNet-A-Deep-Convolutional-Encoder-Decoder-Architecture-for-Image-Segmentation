import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.model.segnet import SegNet
from src.data.dataloader import dataloader

NUM_EPOCHS = 40


def train():
    train_loader, test_loader = dataloader()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SegNet(3, 12).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(NUM_EPOCHS):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        model.train()
        for i, (images, labels) in enumerate(train_loader):
            #print("start training...%s" % i)
            #images, labels = images.to(device), labels.to(device)

            images = torch.tensor(images, dtype=torch.long, device=device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)

            optimizer.zero_grad()
            outputs = model(images.float())
            # print(outputs.shape)
            # loss = criterion(outputs, labels)
            loss = cross_entropy2d(outputs, labels)
            train_loss += loss.item()
            # train_acc += (outputs.max(1)[1] == labels).sum().item()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader.dataset)
        #avg_train_acc = train_acc / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                # images = images.to(device)
                # labels = labels.to(device)

                images = torch.tensor(images, dtype=torch.long, device=device)
                labels = torch.tensor(labels, dtype=torch.long, device=device)

                outputs = model(images.float())
                loss = cross_entropy2d(outputs, labels)
                val_loss += loss.item()
                #val_acc += (outputs.max(1)[1] == labels).sum().item()

        avg_val_loss = val_loss / len(test_loader.dataset)
        #avg_val_acc = val_acc / len(test_loader.dataset)

        print('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}'
              .format(epoch + 1, NUM_EPOCHS, i + 1, loss=avg_train_loss, val_loss=avg_val_loss))

        #print('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'
        #      .format(epoch + 1, NUM_EPOCHS, i + 1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))
        train_loss_list.append(avg_train_loss)
        #train_acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss)
        #val_acc_list.append(avg_val_acc)

    plt.figure()
    plt.plot(range(NUM_EPOCHS), train_loss_list, color='blue', linestyle='-', label='train_loss')
    plt.plot(range(NUM_EPOCHS), val_loss_list, color='green', linestyle='--', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.grid()

    # plt.figure()
    # plt.plot(range(NUM_EPOCHS), train_acc_list, color='blue', linestyle='-', label='train_acc')
    # plt.plot(range(NUM_EPOCHS), val_acc_list, color='green', linestyle='--', label='val_acc')
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('acc')
    # plt.title('Training and validation accuracy')
    # plt.grid()

    # torch.save(model.state_dict(), './trained_model')

    return model

# TODO: Check This Code!
def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss

train()