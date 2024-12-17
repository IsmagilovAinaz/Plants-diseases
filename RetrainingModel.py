# Загрузка необходимых библиотек
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

if __name__ == '__main__':   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    training_set = datasets.ImageFolder(root=r'\train', transform=transform)
    validation_set = datasets.ImageFolder(root=r'\valid', transform=transform)

    train_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(validation_set, batch_size=32, num_workers=4, pin_memory=True)

    # Загрузка предобученной модели
    model = torch.jit.load('plantsDiseasesModel.pth')

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Переобучение модели
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
        if(val_acc >= 0.94):
            break   

         

    # Сохранение обновленной модели
    scripted_model = torch.jit.script(model)
    scripted_model.save('plantsDiseasesModel_updated.pth')
