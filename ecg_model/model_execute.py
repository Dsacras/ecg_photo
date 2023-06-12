from torch.utils.data import DataLoader, random_split
from data import CustomDatasetUrl
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch
from torchvision import models
from model import train_model, evaluate_model

def execute_model():
    imgs_path = './ECGs/'
    csv_path = './scp_codes.csv'
    csv_file = 'scp_codes.csv'

    batch_size = 32
    val_split = 0.2
    test_split = 0.1

    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # dataset = CustomDataset(imgs_path, csv_path, transform, 0, 2048)
    dataset = CustomDatasetUrl(imgs_path, csv_file, transform, 0, 10)
    print(dataset)
    num_samples = len(dataset)
    print(num_samples)
    num_val = int(val_split * num_samples)
    num_test = int(test_split * num_samples)
    num_train = num_samples - num_val - num_test
    print(num_train)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(model, criterion, optimizer, scheduler, train_dataloader, val_dataloader, device, num_epochs=5)

    evaluate_model(model, criterion, test_dataloader, device)

if __name__ == '__main__':
    execute_model()
