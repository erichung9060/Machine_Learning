import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

# HyperParameters
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 32

# Create your CNN model
class CNN(nn.Module):
    # Constructor
    def __init__(self):  # number 0~9
        super(CNN, self).__init__()

        # Our images are Gray-level(single channel), so input channels = 1. We'll apply 12 filters in the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)

        # A second convolutional layer takes 12 input channels, and generates 12 outputs
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)

        # A third convolutional layer takes 12 inputs and generates 24 outputs
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)

        # A drop layer deletes 20% of the features to help prevent overfitting
        self.drop = nn.Dropout2d(p=0.2)

        # We'll apply max pooling with a kernel size of 2
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Our 28x28 image tensors will be pooled twice with a kernel size of 2. 28/2/2 is 7.
        # So our feature tensors are now 7 image 7, and we've generated 24 of them
        # We need to flatten these and feed them to a fully-connected layer
        # to map them to the probability for each class
        self.fc = nn.Linear(in_features=7 * 7 * 24, out_features=10)

    def forward(self, image):
        # Use a relu activation function after layer 1 (convolution 1 and pool)
        image = F.relu(self.pool(self.conv1(image)))

        # Use a relu activation function after layer 2 (convolution 2 and pool)
        image = F.relu(self.pool(self.conv2(image)))

        # Select some features to drop after the 3rd convolution to prevent overfitting
        image = F.relu(self.drop(self.conv3(image)))

        # Only drop the features if this is a training pass
        image = F.dropout(image, training=self.training)

        # Flatten
        image = image.view(-1, 7 * 7 * 24)
        # Feed to fully-connected layer to predict class and use log_softmax to activate
        image = F.log_softmax(self.fc(image), dim=1)
        
        return image

# Create model
model = CNN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()  # loss function
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)  # optimizer

# Train function
def train(model, trainLoader, valLoader, optimizer, criterion, epochs):
    model.train()
    testAccuracy = 0
    bestModel = model
    model_loss = {'train': [], 'val': []}
    model_acc = {'train': [], 'val': []}
    for i in range(epochs):
        totalLoss = 0
        accuracy = 0
        count = 0
        
        for image, label in trainLoader:
            image = image.to(device)
            label = label.to(device, dtype=torch.long)
            optimizer.zero_grad()  # clear previous gradient

            output = model(image)  # sent images into model and do forward propagation

            loss = criterion(output, label)
            _, predicted = torch.max(output.data, 1)
            count += len(image)
            loss.backward()  # according lossto do back propagation，calculate gradient
            optimizer.step()  # gradient descent
            accuracy += (predicted == label).sum().item()
            totalLoss += loss.item()*len(label)
        
        print('Epoch: %d | Train Loss: %.4f | Train Accuracy: %.2f' % (i+1, totalLoss/count, accuracy/count))
        model_loss['train'].append(totalLoss/count)
        model_acc['train'].append(accuracy/count)

        # save model
        if (i % 1 == 0):
            tmpAccuracy = val(model, valLoader, criterion, i+1, model_loss, model_acc)
            if (tmpAccuracy > testAccuracy):
                testAccuracy = tmpAccuracy
                bestModel = model
                epoch = i
                torch.save(bestModel, "./checkpoint/"+str(epoch)+"_"+str(testAccuracy)+".pth")

    torch.save(bestModel, "./checkpoint/final_model.pth")
    return model_loss, model_acc

# Validation function
def val(model, valLoader, criterion, epoch, model_loss, model_acc):
    model.eval()
    totalLoss = 0
    accuracy = 0
    count = 0
    for image, label in valLoader:
        image = image.to(device)
        label = label.to(device, dtype=torch.long)
        output = model(image)
        loss = criterion(output, label)
        _, predicted = torch.max(output.data, 1)
        count += len(image)
        accuracy += (predicted == label).sum().item()
        totalLoss += loss.item()*len(label)
    
    print('Epoch: %d | Val   Loss: %.4f | Val   Accuracy: %.2f'% (epoch, totalLoss/count, accuracy/count))
    if epoch == NUM_EPOCHS: print("----------------------------------------------------------------\n")
    model_loss['val'].append(totalLoss/count)
    model_acc['val'].append(accuracy/count)
    return (accuracy / count)

# Predict function
def test(Test_DataLoader, model, criterion):
    model.eval()
    true = []  # truly label
    pred = []  # predicted label
    with torch.no_grad():
        for images, target in Test_DataLoader:
            images = images.to(device)
            target = target.to(device, dtype=torch.long)
            output = model(images)
            _, preds = torch.max(output.data, 1)

            pred.append(preds.item())
            # y_pred.extend(preds.cpu().view(-1).numpy())
            true.extend([target.cpu().view(-1).numpy()][0])
            # print(preds.item())

    return true, pred


def main():
    # download data
    Train_Data = torchvision.datasets.MNIST( root='MINST', train=True, download=True, transform=torchvision.transforms.ToTensor())
    Test_Data = torchvision.datasets.MNIST( root='MINST', train=False, download=True, transform=torchvision.transforms.ToTensor())
    Train_Subset, Val_Subset = torch.utils.data.random_split( Train_Data, [48000, 12000], generator=torch.Generator().manual_seed(1))

    # load data
    Train_DataLoader = torch.utils.data.DataLoader( Train_Subset, batch_size=BATCH_SIZE, shuffle=True)
    Val_DadaLoader = torch.utils.data.DataLoader( Val_Subset, batch_size=BATCH_SIZE, shuffle=True)
    Test_DataLoader = torch.utils.data.DataLoader( Test_Data, batch_size=1, shuffle=False)  

    # print number of dataset
    print(f"Train data size:\t{len(Train_Subset)}\nValidation data size:\t{len(Val_Subset)}\nTest data size:\t\t{len(Test_Data)}\n")
    
    # show images
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.imshow(Train_Data.data[i].numpy(), cmap='gray')
        plt.title('%i' % Train_Data.targets[i])
    plt.pause(2)
    plt.close()

    # show model summary
    summary(model, (1, 28, 28), BATCH_SIZE)

    # start training
    model_loss, model_acc = train(model, Train_DataLoader, Val_DadaLoader, optimizer, criterion, NUM_EPOCHS)
    
    # print loss curve
    plt.plot(model_loss['train'], label='train loss')
    plt.plot(model_loss['val'], label='val loss')
    plt.title("Train and Val Loss")
    plt.pause(2)
    plt.close()

    # print accuracy curve
    plt.plot(model_acc['train'], label='train accuracy')
    plt.plot(model_acc['val'], label='val accuracy')
    plt.title("Train and Val Accuracy")
    plt.legend()
    plt.pause(2)
    plt.close()

    # start predicting
    true, pred = test(Test_DataLoader, model, criterion)

    # show predicted result
    for i in range(15):
        plt.subplot(3, 5, i+1)
        plt.imshow(Test_Data.data[i].numpy(), cmap='gray')
        plt.title(str(pred[i])) 
        plt.subplots_adjust(wspace=1.0, hspace=1.0)
    plt.pause(5)
    plt.close()

    # print Confusion Matrix
    cf_matrix = confusion_matrix(true, pred)
    df_cm = pd.DataFrame(cf_matrix)
    plt.title('confusion matrix')
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='OrRd')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.show()


if __name__ == '__main__':
    main()