
from __future__ import print_function, division
import os
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
from torch import no_grad

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


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
       

        #return _img, _target
        #return self.transform(sample)
        _img = self.transform(_img)
        _target = self.transform_label(from_numpy(_target).long())
        sample = {'image': _img, 'label': _target}
        return sample

        ## label use from_numpy
        ##


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = np.array(Image.open(self.categories[index]))

        return _img, _target

    def transform(self, sample):
        composed_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(400),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        return composed_transforms(sample)

    def transform_label(self, sample):
        composed_transforms = transforms.Compose([
            transforms.CenterCrop(400)])

        return composed_transforms(sample)

    

#     def __str__(self):
#         return 'VOC2012(split=' + str(self.split) + ')'

# def decode_segmap(label_mask, dataset, plot=False):
#     label_colours = get_pascal_labels()
#     r = label_mask.copy()
#     g = label_mask.copy()
#     b = label_mask.copy()
#     for ll in range(0, 21):
#         r[label_mask == ll] = label_colours[ll, 0]
#         g[label_mask == ll] = label_colours[ll, 1]
#         b[label_mask == ll] = label_colours[ll, 2]
#     rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
#     rgb[:, :, 0] = r / 255.0
#     rgb[:, :, 1] = g / 255.0
#     rgb[:, :, 2] = b / 255.0

# def get_pascal_labels():
#     return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
#                        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
#                        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
#                        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
#                        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
#                        [0, 64, 128]])

"""Setting up an untrained ResNet50 model """

#torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

#model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=False)
model = fcn_resnet50(pretrained=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

#!nvidia-smi



# import gc

# # Clean up model in-case of memory cap (debugging)
# del model
# gc.collect()
# torch.cuda.empty_cache()

"""Setting up the Training, Validation, and Testing datasets"""

training = VOCSegmentation(split='train')
full_validation = VOCSegmentation(split='val')

validation, test = torch.utils.data.random_split(full_validation, [82,81 ], generator=torch.Generator().manual_seed(10))

batch = 2

train_dataloader = DataLoader(training, shuffle=True, batch_size=batch, num_workers=1)
val_dataloader = DataLoader(validation, shuffle=False, batch_size=batch, num_workers=1)
test_dataloader = DataLoader(test, shuffle=False, batch_size=batch, num_workers=1)

"""Training function"""

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/VOC_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 10

optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
# scaler = torch.cuda.amp.GradScaler() 

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_dataloader):

        if (torch.cuda.is_available()):
          inputs = data['image'].cuda()
          labels = data['label'].cuda()
        else:
          inputs = data['image']
          labels = data['label']

        optimizer.zero_grad()
        # print('size ', inputs.size())
        # print('labels: ',labels.size())
        # print('test')

        model.train()

        outputs = model(inputs)['out']
        
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        # running_loss += loss.item()
        # if i % 1000 == 999:
        #     last_loss = running_loss / 1000 # loss per batch
        #     print('  batch {} loss: {}'.format(i + 1, last_loss))
        #     tb_x = epoch_index * len(train_dataloader) + i + 1
        #     tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        #     running_loss = 0.

    return last_loss

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)
    # model.train(False)

    running_vloss = 0.0
    model.eval()
    for i, vdata in enumerate(val_dataloader):
        if (torch.cuda.is_available()):
          vinputs = vdata['image'].cuda()
          vlabels = vdata['label'].cuda()
        else:
          vinputs = vdata['image']
          vlabels = vdata['label']
        with no_grad():
          voutputs = model(vinputs)['out']
        vloss = criterion(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

     #Log the running loss averaged per batch
     #for both training and validation
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
