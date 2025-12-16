import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def data_loader(train_dir, valid_dir, batch_size, test=False, shuffle=True):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    transform = transforms.Compose([transforms.Resize((227,227)),transforms.ToTensor(), normalize])
    if test:
        dataset = datasets.ImageFolder(root=train_dir, transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader
    
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)
    
    # Attach labels to datasets according to their folder names
    train_dataset.classes = [d.name for d in os.scandir(train_dir) if d.is_dir()]
    valid_dataset.classes = [d.name for d in os.scandir(valid_dir) if d.is_dir()]

    # Debugging statements to check labels
    print(f'Train dataset classes: {train_dataset.classes}')
    print(f'Valid dataset classes: {valid_dataset.classes}')

    num_train = len(train_dataset)
    num_valid = len(valid_dataset)
    train_indices = list(range(num_train))
    valid_indices = list(range(num_valid))
    print(f'Number of training samples: {num_train}')
    print(f'Number of validation samples: {num_valid}')
    
    # if shuffle:
    #     np.random.seed(random_seed)
    #     np.random.shuffle(train_indices)
    #     np.random.shuffle(valid_indices)
    
    # train_sampler = SubsetRandomSampler(train_indices)
    # valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)

    return (train_loader, valid_loader)

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(  
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
num_epochs = 20
batch_size = 8
learning_rate = 0.005
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps")

train_loader, valid_loader = data_loader(train_dir='train', valid_dir='valid', batch_size=batch_size, shuffle=True)
# test_loader = data_loader(train_dir='test_train', valid_dir = 'valid', batch_size=batch_size, test=True)

# Ensure num_classes matches the number of classes in the dataset
num_classes = len(train_loader.dataset.classes)
print(f'Number of classes: {num_classes}')


# Update model initialization
model = VGG16(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)

total_step = len(train_loader)
class_names = ["Other", "Recyclable", "Harmful", "Kitchen"]
# training loop
print("total_steps:", total_step)
# quit()
for epoch in range(num_epochs):
    # Add tqdm progress bar
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", total=len(train_loader))):  
        # Take the Tensors onto the device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Convert the tensor image to numpy array and transpose to (H, W, C)
        # print(outputs[0].argmax().item())
        # npimg = images[0].cpu().numpy().transpose((1, 2, 0))
        # plt.imshow(npimg)
        # plt.title(f'Predicted: {outputs[0].argmax().item()}, Actual: {labels[0].item()}')
        # plt.show()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Predicted: {class_names[outputs[0].argmax().item()]}, Actual: {class_names[labels[0].item()]}')

    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            del images, labels, outputs

        print('Accuracy of the network on the validation images: {} %'.format(100 * correct / total))
    
    # Save the model checkpoint every 50 epochs
    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')