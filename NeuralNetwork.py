import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        """self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4)
        )
        self.res1 = nn.Sequential(nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ), nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4)
        )"""
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        """self.res2 = nn.Sequential(nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        ), nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        ))
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, 38))    #38 - колво классов"""

        self.maxpool = nn.MaxPool2d(4)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 4096) 
        self.dropout1 = nn.Dropout(0.15)
        #self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(4096, 38)

    def forward(self, x):
        """x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x) + x
        x = self.classifier(x)"""
        x = self.conv_layers(x)
        x = self.maxpool(x)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

if __name__ == '__main__':     
    flag = False
    if(flag):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        

        transform = transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        # замените на ваши пути к данным
        training_set = datasets.ImageFolder(root=r'path\train', transform=transform)
        validation_set = datasets.ImageFolder(root=r'path\valid', transform=transform)

        train_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(validation_set, batch_size=32, num_workers=4, pin_memory=True)

        model = CNN().to(device)
        #model = to_device(CNN(), device)
        #model.load_state_dict(torch.load('model2.pth', map_location=device))
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()


        # Обучение модели
        num_epochs = 5
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


            train_loss = running_loss / len(train_loader)
            train_acc = correct / total

            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for images_val, labels_val in val_loader:
                    images_val, labels_val = images_val.to(device), labels_val.to(device)
                    outputs_val = model(images_val)
                    loss_val = criterion(outputs_val, labels_val)

                    val_loss += loss_val.item()
                    _, predicted_val = torch.max(outputs_val.data, 1)
                    total_val += labels_val.size(0)
                    correct_val += (predicted_val == labels_val).sum().item()

            val_loss /= len(val_loader)
            val_acc = correct_val / total_val

            print(f'Epoch [{epoch+1}/{num_epochs}], '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
        torch.save(model.state_dict(), 'modelRenew3Plants.pth')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #model1 = torch.load('model2.pth', weights_only=False)
        #torch.save(model1.state_dict(), 'model1.pth')
        model = CNN().to(device)
        model.load_state_dict(torch.load('modelNew.pth', map_location=device))
        #model = torch.load('model.pth', weights_only=False)
        scripted_model = torch.jit.script(model)
        print(model)
        scripted_model.save('plantsDiseasesModelNew.pth')
