import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=False)

comp = transforms.Compose([
    transforms.CenterCrop(512),
    transforms.ToTensor()
])

labels = {
1:'aeroplane',
2:'bicycle',
3:'bird',
4:'boat',
5:'bottle',
6:'bus',
7:'car',
8:'cat',
9:'chair',
10:'cow',
11:'diningtable',
12:'dog',
13:'horse',
14:'motorbike',
15:'person',
16:'pottedplant',
17:'sheep',
18:'sofa',
19:'train',
20:'tvmonitor'
}

training = datasets.VOCSegmentation(
    root='data/train',
    year='2012',
    image_set='train',
    download=True,
    transform=comp
)

train_masks = training.masks

print("Training Set Size: ", len(training.images))

full_validation = datasets.VOCSegmentation(
    root='data/fullval',
    year='2012',
    image_set='val',
    download=True,
    transform=comp
)

print("Full Validation Set Size: ", len(full_validation.images))

validation, test = torch.utils.data.random_split(full_validation, [724, 725], generator=torch.Generator().manual_seed(10))

print("Validation Set Size: ", len(validation))

print("Test Set Size: ", len(test))

batch = 4

train_dataloader = DataLoader(training, shuffle=False, batch_size=batch, num_workers=2)
val_dataloader = DataLoader(validation, shuffle=False, batch_size=batch, num_workers=2)
test_dataloader = DataLoader(test, shuffle=False, batch_size=batch, num_workers=2)

train_sampler = torch.utils.data.RandomSampler(training)
test_sampler = torch.utils.data.SequentialSampler(test)

from torch import utils

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()
current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=2, cooldown=2)
scaler = torch.cuda.amp.GradScaler() 

def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    for image, target in metric_logger.log_every(data_loader, 10, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        
epoch = 10

for epoch in range(epoch):
        train_one_epoch(model, criterion, optimizer, train_dataloader, lr_scheduler, 'cuda', epoch, scaler)
