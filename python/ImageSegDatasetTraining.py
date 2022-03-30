import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=False)

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
    transform=ToTensor()
)

fullvalidation = datasets.VOCSegmentation(
    root='data',
    year='2012',
    image_set='val',
    download=True,
    transform=ToTensor()
)

validation, test = torch.utils.data.random_split(fullvalidation, [1,449/2, 1,449], generator=torch.Generator().manual_seed(10))
