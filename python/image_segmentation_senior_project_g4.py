! pip install "torchmetrics"

from __future__ import print_function, division
import os
import collections
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import utils
from torch import from_numpy
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import fcn_resnet50
from torchvision import datasets

# Downloads VOC data

datasets.VOCSegmentation(
    root='VOC',
    year='2012',
    image_set='train',
    download=True,
)

# VOC Segmentation Dataset handler
class VOCSegmentation(Dataset):
    NUM_CLASSES = 21

    def __init__(self, base_dir='/content/VOC/VOCdevkit/VOC2012', split='train'):
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = '/content/VOC/VOCdevkit/VOC2012/JPEGImages'
        self._cat_dir = '/content/VOC/VOCdevkit/VOC2012/SegmentationClass'

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        _splits_dir = '/content/VOC/VOCdevkit/VOC2012/ImageSets/Segmentation'

        self.im_ids = []
        self.images = []
        self.categories = []
        

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index: int):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])
       
        _img = self.transform(_img)
        _target = self.transform_label(_target)
        return _img, _target

    def transform(self, sample):
        composed_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(400),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        return composed_transforms(sample)

    def transform_label(self, sample):
        composed_transforms = transforms.Compose([
            transforms.CenterCrop(400),
            transforms.ToTensor()])

        return composed_transforms(sample)

# find the share link of the file/folder on Google Drive
# !gdown --folder 1zGsz0hy26OBg1gAX_loyYrW6DEKLjB2k

"""Setting up an untrained ResNet50 model """

model = deeplabv3_resnet50(pretrained=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

"""Setting up the Training, Validation, and Testing datasets"""

training = VOCSegmentation(split='train')
full_validation = VOCSegmentation(split='val')

# Loads only for specific val.txt provided
validation, test = torch.utils.data.random_split(full_validation, [82, 81], generator=torch.Generator().manual_seed(10))

batch = 4

train_dataloader = DataLoader(training, shuffle=True, batch_size=batch)
val_dataloader = DataLoader(validation, shuffle=False, batch_size=batch)
test_dataloader = DataLoader(test, shuffle=False, batch_size=batch)

# Training function

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/VOC_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 40

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
best_vloss = -999
losses = []

model.to(device)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)['out']      
        labels = torch.argmax(labels, dim=1)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / 21

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)
    model.train(False)

    losses.append(avg_loss)

    optimizer.zero_grad()

    running_vloss = 0.0
    for i, vdata in enumerate(val_dataloader):
        vinputs, vlabels = vdata

        voutputs = model(vinputs.to(device))['out']

        vlabels = torch.argmax(vlabels, dim=1)

        vloss = criterion(voutputs, vlabels.to(device))
        running_vloss += vloss.item()

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

plt.xlim([2, 40])
plt.ylim([0, .04])
plt.plot(losses)

from torch.utils.mobile_optimizer import optimize_for_mobile

scripted_module = torch.jit.script(model)
optimized_scripted_module = optimize_for_mobile(scripted_module)

# Export full jit version model (not compatible with lite interpreter)
scripted_module.save("deeplabv3_scripted.pt")
# Export lite interpreter version model (compatible with lite interpreter)
scripted_module._save_for_lite_interpreter("deeplabv3_scripted.ptl")
# Export to optimized lite interpreter model
optimized_scripted_module._save_for_lite_interpreter("deeplabv3_correct_lr.ptl")

model.to('cpu')

def IoU(label, pred, num_classes=21):        
    pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = label.view(-1)
    
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else: 
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)

total_tacc = 0.0
with torch.no_grad():
  for i, tdata in enumerate(test_dataloader):
      tinputs, tlabels = tdata

      toutputs = model(tinputs.to('cpu'))['out']

      tacc = IoU(tlabels, toutputs)

      print('Class ID', i, 'Accuracy: ', 1 - tacc)

      total_tacc += (1 - tacc.item())

total_tacc = total_tacc / 21

print('Average semantic accuracy: ', total_tacc)