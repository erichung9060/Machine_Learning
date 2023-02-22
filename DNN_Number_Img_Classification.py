import time
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

"""## Setting Hyperparamters"""
NUM_EPOCHS = 20
BTACH_SIZE = 50
LEARNING_RATE = 1e-3

"""## DownLoad Dataset """
train_data = torchvision.datasets.MNIST( root='MNIST', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST( root='MNIST', train=False, download=True, transform=torchvision.transforms.ToTensor())

"""## Plot DataSet"""
for i in range(5):
    plt.imshow(train_data.data[i].numpy(), cmap='gray')
    plt.title('%i' % train_data.targets[i])
    plt.pause(0.5)

plt.close()

"""## Split Training set into validation and training set"""
train_subset, val_subset = torch.utils.data.random_split(train_data, [1000, 59000], generator=torch.Generator().manual_seed(1))

"""## Load Trainingset to DataLoader """
train_dataloader = DataLoader(train_subset, batch_size=BTACH_SIZE, shuffle=True)
valid_dataloader = DataLoader(val_subset, batch_size=BTACH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

"""## Design Model"""
class DNNMODEL(nn.Module):
    def __init__(self):
        super(DNNMODEL, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10)
        )

    def forward(self, x):
        x = x.view(-1, 28*28) #flatten
        out = self.net(x) #feed into neural network
        out = F.log_softmax(out, dim=1) #activation function of output layer
        return out

loss_func = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DNNMODEL()
model = model.to(device)
optimization = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

"""## Training Progress"""
def train(model, trainloader, valloader, optimizer, criterion):
    model.train() #tell model start to train
    model_loss_record = {
        'train': [],
        'val': [],
    }
    for epoch in range(NUM_EPOCHS):
        #Run Train Data
        train_loss = 0
        train_correct = 0
        for step, (data, targets) in enumerate(trainloader):  # batch
            data, targets = data.to(device), targets.to(device)
            data.requires_grad_()  # need to calculate gradient
            optimizer.zero_grad()  # initialized gradient to 0

            output = model(data)  # compute output

            loss = criterion(output, targets)  # compute loss
            loss.backward()  # back propagtion and compute gradient
            optimizer.step()  # update parameter (based on gradient)
            train_loss += loss.item()*data.size(0) # calculate train loss
            _, predicted = torch.max(output.data, 1) # get predicted number
            train_correct += (predicted == targets).sum().item()  # calculate correct number
        
        trainloss = train_loss/len(trainloader.dataset)
        trainacc = train_correct/len(trainloader.dataset)

        #Run Validation Data
        val_loss = 0
        val_correct = 0
        for step, (data, targets) in enumerate(valloader):
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            loss = criterion(output, targets)
            val_loss += loss.item()*data.size(0)
            _, predicted = torch.max(output.data, 1)
            val_correct += (predicted == targets).sum().item()
        
        valloss = val_loss/len(valloader.dataset)
        valacc = val_correct/len(valloader.dataset)
        
        print(f'running epoch: {epoch+1}')
        print(f'Training Loss  : {trainloss:.4f}\t\tTraining Accuracy  : {trainacc*100:.2f}%')
        print(f'Valid Loss  : {valloss:.4f}\t\tValid Accuracy  : {valacc*100:.2f}%')

        model_loss_record['train'].append(trainloss)
        model_loss_record['val'].append(valloss)
    
    return model_loss_record


model_loss_record = train(model, train_dataloader, valid_dataloader, optimization, loss_func)

"""## Testing Progress"""
def test(model, testloader, criterion):
    model.eval()
    test_loss = 0
    test_correct = 0
    pred = []
    with torch.no_grad():
        for data, targets in testloader:
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            loss = criterion(output, targets)
            test_loss += loss.item()*data.size(0)
            _, predicted = torch.max(output.data, 1)
            test_correct += (predicted == targets).sum().item()
            pred.append(predicted.item())
        
    return test_loss/len(testloader.dataset), test_correct/len(testloader.dataset), pred

_, _, pred = test(model, test_dataloader, criterion=loss_func)

"""## Plot Predicted"""
for i in range(10):
    plt.imshow(test_dataset.data[i].numpy(), cmap='gray')
    plt.title('predicted ' + str(pred[i]))
    plt.pause(0.5)
plt.close()

"""## Plot Result"""
plt.figure("Loss Figure")
plt.plot(model_loss_record['train'])
plt.plot(model_loss_record['val'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# # """## Save Trained Model"""
# # torch.save(model, "numer_classification")

# # """## Load Pretrained Model"""
# # model = torch.load("numer_classification")