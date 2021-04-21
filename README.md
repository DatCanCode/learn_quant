```python
torch.__file__
```




    '/opt/conda/envs/quant/lib/python3.8/site-packages/torch/__init__.py'




```python
import torchvision
import torch
from torchvision import transforms
import time
import copy
from tqdm import tqdm
```


```python
SEED = 25
torch.manual_seed(SEED)
import numpy as np
np.random.seed(SEED)
import random
random.seed(SEED)
```


```python
data_dir = '~/.torch/data'

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


img_datasets = {
    'train': torchvision.datasets.CIFAR10(data_dir, train=True,
                                          transform=data_transforms['train'], download=True),
    'val': torchvision.datasets.CIFAR10(data_dir, train=False,
                                          transform=data_transforms['val'], download=True)}

dataloaders = {x: torch.utils.data.DataLoader(img_datasets[x], batch_size=32,
                                              shuffle=True, num_workers=32)
              for x in ['train', 'val']}

dataset_sizes = {x: len(dataloaders[x]) for x in dataloaders.keys()}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

    Files already downloaded and verified
    Files already downloaded and verified



```python
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
  """
  Support function for model training.

  Args:
    model: Model to be trained
    criterion: Optimization criterion (loss)
    optimizer: Optimizer to use for training
    scheduler: Instance of ``torch.optim.lr_scheduler``
    num_epochs: Number of epochs
    device: Device to run the training on. Must be 'cpu' or 'cuda'
  """
  since = time.time()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()  # Set model to training mode
      else:
        model.eval()   # Set model to evaluate mode

      running_loss = 0.0
      running_corrects = 0

      # Iterate over data.
      for inputs, labels in tqdm(dataloaders[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)

          # backward + optimize only if in training phase
          if phase == 'train':
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
      if phase == 'train':
        scheduler.step()

      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double() / dataset_sizes[phase]

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        phase, epoch_loss, epoch_acc))

      # deep copy the model
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    print()

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  # load best model weights
  model.load_state_dict(best_model_wts)
  return model
```


```python
import torch.optim as optim
import torch.nn as nn

device
```




    device(type='cuda', index=0)




```python
model_float = torchvision.models.mobilenet_v2(pretrained=True)

for params in model_float.parameters():
    params.requires_grad = False

model_float.classifier[1] = nn.Linear(model_float.classifier[1].in_features, len(img_datasets['val'].classes))

model_float = model_float.to(device)


criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_float.parameters(), lr=0.01, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
```


```python
model_float = train_model(model_float, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10, device=device)
```

      0%|          | 0/1563 [00:00<?, ?it/s]

    Epoch 0/9


    100%|██████████| 1563/1563 [00:39<00:00, 39.76it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 31.8268 Acc: 21.6705


    100%|██████████| 313/313 [00:10<00:00, 30.77it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 26.7346 Acc: 23.3738
    
    Epoch 1/9


    100%|██████████| 1563/1563 [00:40<00:00, 38.98it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 30.8044 Acc: 22.4613


    100%|██████████| 313/313 [00:09<00:00, 33.02it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 31.3225 Acc: 22.6486
    
    Epoch 2/9


    100%|██████████| 1563/1563 [00:40<00:00, 38.76it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 30.9927 Acc: 22.5457


    100%|██████████| 313/313 [00:09<00:00, 32.83it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 26.1890 Acc: 23.6709
    
    Epoch 3/9


    100%|██████████| 1563/1563 [00:39<00:00, 39.44it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 30.7973 Acc: 22.4946


    100%|██████████| 313/313 [00:09<00:00, 32.98it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 28.1808 Acc: 23.1789
    
    Epoch 4/9


    100%|██████████| 1563/1563 [00:39<00:00, 39.10it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 30.8304 Acc: 22.6468


    100%|██████████| 313/313 [00:09<00:00, 32.65it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 24.3661 Acc: 24.2236
    
    Epoch 5/9


    100%|██████████| 1563/1563 [00:39<00:00, 39.09it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 30.7493 Acc: 22.7038


    100%|██████████| 313/313 [00:09<00:00, 32.80it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 28.7463 Acc: 22.9681
    
    Epoch 6/9


    100%|██████████| 1563/1563 [00:39<00:00, 39.60it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 31.1733 Acc: 22.7057


    100%|██████████| 313/313 [00:09<00:00, 33.63it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 25.9346 Acc: 23.8275
    
    Epoch 7/9


    100%|██████████| 1563/1563 [00:39<00:00, 39.18it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 25.0437 Acc: 23.8714


    100%|██████████| 313/313 [00:09<00:00, 33.09it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 20.2349 Acc: 24.9808
    
    Epoch 8/9


    100%|██████████| 1563/1563 [00:39<00:00, 39.58it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 23.9762 Acc: 23.9475


    100%|██████████| 313/313 [00:09<00:00, 34.06it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 19.9328 Acc: 24.9808
    
    Epoch 9/9


    100%|██████████| 1563/1563 [00:39<00:00, 39.55it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 23.6106 Acc: 23.8996


    100%|██████████| 313/313 [00:09<00:00, 33.25it/s]

    val Loss: 19.3941 Acc: 25.2588
    
    Training complete in 8m 13s
    Best val Acc: 25.258786


    



```python
params = list(model_float.parameters())
len(params)
```




    158




```python
for param in params[70:]:
    param.requires_grad = True
```


```python
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_float.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
```


```python
model_float = train_model(model_float, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=15, device=device)
```

      0%|          | 0/1563 [00:00<?, ?it/s]

    Epoch 0/14


    100%|██████████| 1563/1563 [00:55<00:00, 28.29it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 13.7838 Acc: 27.4632


    100%|██████████| 313/313 [00:09<00:00, 33.17it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 8.1108 Acc: 29.2173
    
    Epoch 1/14


    100%|██████████| 1563/1563 [00:55<00:00, 28.19it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 6.6823 Acc: 29.6833


    100%|██████████| 313/313 [00:09<00:00, 33.26it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 7.2202 Acc: 29.6550
    
    Epoch 2/14


    100%|██████████| 1563/1563 [00:55<00:00, 28.29it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 4.1760 Acc: 30.5323


    100%|██████████| 313/313 [00:09<00:00, 32.51it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 7.6660 Acc: 29.5974
    
    Epoch 3/14


    100%|██████████| 1563/1563 [00:55<00:00, 28.08it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 2.8644 Acc: 30.9699


    100%|██████████| 313/313 [00:09<00:00, 32.84it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 7.4314 Acc: 29.7252
    
    Epoch 4/14


    100%|██████████| 1563/1563 [00:55<00:00, 27.98it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 1.9112 Acc: 31.2994


    100%|██████████| 313/313 [00:09<00:00, 32.94it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 7.5025 Acc: 29.8946
    
    Epoch 5/14


    100%|██████████| 1563/1563 [00:55<00:00, 28.14it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 1.3783 Acc: 31.5138


    100%|██████████| 313/313 [00:09<00:00, 32.55it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 7.7162 Acc: 29.8978
    
    Epoch 6/14


    100%|██████████| 1563/1563 [00:55<00:00, 28.18it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 1.1685 Acc: 31.5982


    100%|██████████| 313/313 [00:09<00:00, 32.85it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 7.3491 Acc: 29.9904
    
    Epoch 7/14


    100%|██████████| 1563/1563 [00:54<00:00, 28.44it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 0.6166 Acc: 31.7933


    100%|██████████| 313/313 [00:09<00:00, 32.53it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 7.0850 Acc: 30.0447
    
    Epoch 8/14


    100%|██████████| 1563/1563 [00:53<00:00, 29.06it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 0.4317 Acc: 31.8676


    100%|██████████| 313/313 [00:09<00:00, 32.66it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 7.0728 Acc: 30.0703
    
    Epoch 9/14


    100%|██████████| 1563/1563 [00:53<00:00, 29.40it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 0.3486 Acc: 31.8996


    100%|██████████| 313/313 [00:09<00:00, 32.31it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 6.9423 Acc: 30.1278
    
    Epoch 10/14


    100%|██████████| 1563/1563 [00:53<00:00, 29.16it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 0.3330 Acc: 31.9060


    100%|██████████| 313/313 [00:09<00:00, 33.08it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 6.9015 Acc: 30.1182
    
    Epoch 11/14


    100%|██████████| 1563/1563 [00:53<00:00, 29.26it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 0.2680 Acc: 31.9283


    100%|██████████| 313/313 [00:09<00:00, 33.84it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 6.9146 Acc: 30.1310
    
    Epoch 12/14


    100%|██████████| 1563/1563 [00:53<00:00, 29.00it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 0.3022 Acc: 31.9136


    100%|██████████| 313/313 [00:10<00:00, 31.23it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 6.9332 Acc: 30.1342
    
    Epoch 13/14


    100%|██████████| 1563/1563 [00:54<00:00, 28.78it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 0.2586 Acc: 31.9277


    100%|██████████| 313/313 [00:09<00:00, 32.01it/s]
      0%|          | 0/1563 [00:00<?, ?it/s]

    val Loss: 6.9605 Acc: 30.1502
    
    Epoch 14/14


    100%|██████████| 1563/1563 [00:54<00:00, 28.55it/s]
      0%|          | 0/313 [00:00<?, ?it/s]

    train Loss: 0.2431 Acc: 31.9360


    100%|██████████| 313/313 [00:09<00:00, 31.33it/s]

    val Loss: 6.9848 Acc: 30.1278
    
    Training complete in 16m 5s
    Best val Acc: 30.150160


    



```python
torch.save(model_float.state_dict(), "mobilenetv2_cifar10.pth")
```


```python
from torch.quantization import QuantStub, DeQuantStub

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            # Replace with ReLU
            nn.ReLU(inplace=False)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup, momentum=0.1),
        ])
        self.conv = nn.Sequential(*layers)
        # Replace torch.add with floatfunctional
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        x = self.quant(x)

        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)
```


```python
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5

def load_model(model_file):
    model = MobileNetV2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
```


```python
data_path = '~/.torch/data'
saved_model_dir = './'
float_model_file = 'mobilenetv2_cifar10.pth'
scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'

train_batch_size = 32
eval_batch_size = 50

data_loader = dataloaders['train']

data_loader_test = torch.utils.data.DataLoader(img_datasets['val'], batch_size=eval_batch_size, num_workers=32)

criterion = nn.CrossEntropyLoss()
float_model = load_model(saved_model_dir + float_model_file).to('cpu')

# Next, we'll "fuse modules"; this can both make the model faster by saving on memory access
# while also improving numerical accuracy. While this can be used with any model, this is
# especially common with quantized models.

print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv)
float_model.eval()

# Fuses modules
float_model.fuse_model()

# Note fusion of Conv+BN+Relu and Conv+Relu
print('\n Inverted Residual Block: After fusion\n\n',float_model.features[1].conv)
```

    
     Inverted Residual Block: Before fusion 
    
     Sequential(
      (0): ConvBNReLU(
        (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (1): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    
     Inverted Residual Block: After fusion
    
     Sequential(
      (0): ConvBNReLU(
        (0): ConvReLU2d(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32)
          (1): ReLU()
        )
        (1): Identity()
        (2): Identity()
      )
      (1): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
      (2): Identity()
    )



```python
import os
num_eval_batches = 200

print("Size of baseline model")
print_size_of_model(float_model)

top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)
```

    Size of baseline model
    Size (MB): 8.926889
    ........................................................................................................................................................................................................Evaluation accuracy on 10000 images, 94.43



```python
num_calibration_batches = 32

myModel = load_model(saved_model_dir + float_model_file).to('cpu')
myModel.eval()

# Fuse Conv, bn and relu
myModel.fuse_model()

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
myModel.qconfig = torch.quantization.default_qconfig
print(myModel.qconfig)
torch.quantization.prepare(myModel, inplace=True)

# Calibrate first
print('Post Training Quantization Prepare: Inserting Observers')
print('\n Inverted Residual Block:After observer insertion \n\n', myModel.features[1].conv)

# Calibrate with the training set
evaluate(myModel, criterion, data_loader, neval_batches=num_calibration_batches)
print('Post Training Quantization: Calibration done')

# Convert to quantized model
torch.quantization.convert(myModel, inplace=True)
print('Post Training Quantization: Convert done')
print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',myModel.features[1].conv)

print("Size of model after quantization")
print_size_of_model(myModel)

top1, top5 = evaluate(myModel, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
```

    QConfig(activation=functools.partial(<class 'torch.quantization.observer.MinMaxObserver'>, reduce_range=True), weight=functools.partial(<class 'torch.quantization.observer.MinMaxObserver'>, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
    Post Training Quantization Prepare: Inserting Observers
    
     Inverted Residual Block:After observer insertion 
    
     Sequential(
      (0): ConvBNReLU(
        (0): ConvReLU2d(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32)
          (1): ReLU()
          (activation_post_process): MinMaxObserver(min_val=inf, max_val=-inf)
        )
        (1): Identity()
        (2): Identity()
      )
      (1): Conv2d(
        32, 16, kernel_size=(1, 1), stride=(1, 1)
        (activation_post_process): MinMaxObserver(min_val=inf, max_val=-inf)
      )
      (2): Identity()
    )


    /opt/conda/envs/quant/lib/python3.8/site-packages/torch/quantization/observer.py:121: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
      warnings.warn(


    ................................Post Training Quantization: Calibration done


    /opt/conda/envs/quant/lib/python3.8/site-packages/torch/quantization/observer.py:243: UserWarning: must run observer before calling calculate_qparams.                                        Returning default scale and zero point 
      warnings.warn(


    Post Training Quantization: Convert done
    
     Inverted Residual Block: After fusion and quantization, note fused modules: 
    
     Sequential(
      (0): ConvBNReLU(
        (0): QuantizedConvReLU2d(32, 32, kernel_size=(3, 3), stride=(1, 1), scale=0.1202375590801239, zero_point=0, padding=(1, 1), groups=32)
        (1): Identity()
        (2): Identity()
      )
      (1): QuantizedConv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), scale=0.18784166872501373, zero_point=63)
      (2): Identity()
    )
    Size of model after quantization
    Size (MB): 2.360679
    ........................................................................................................................................................................................................Evaluation accuracy on 10000 images, 48.33



```python
per_channel_quantized_model = load_model(saved_model_dir + float_model_file)
per_channel_quantized_model.eval()
per_channel_quantized_model.fuse_model()
per_channel_quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(per_channel_quantized_model.qconfig)

torch.quantization.prepare(per_channel_quantized_model, inplace=True)
evaluate(per_channel_quantized_model,criterion, data_loader, num_calibration_batches)
torch.quantization.convert(per_channel_quantized_model, inplace=True)
top1, top5 = evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
torch.jit.save(torch.jit.script(per_channel_quantized_model), saved_model_dir + scripted_quantized_model_file)
```

    QConfig(activation=functools.partial(<class 'torch.quantization.observer.HistogramObserver'>, reduce_range=True), weight=functools.partial(<class 'torch.quantization.observer.PerChannelMinMaxObserver'>, dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
    ................................

    /opt/conda/envs/quant/lib/python3.8/site-packages/torch/quantization/observer.py:955: UserWarning: must run observer before calling calculate_qparams.                                    Returning default scale and zero point 
      warnings.warn(


    ........................................................................................................................................................................................................Evaluation accuracy on 10000 images, 80.12



```python

```


```python
def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end = '')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('Loss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=top1, top5=top5))
    return

```


```python
qat_model = load_model(saved_model_dir + float_model_file)
qat_model.fuse_model()

optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
```


```python
torch.quantization.prepare_qat(qat_model, inplace=True)
print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n',qat_model.features[1].conv)

```

    Inverted Residual Block: After preparation for QAT, note fake-quantization modules 
     Sequential(
      (0): ConvBNReLU(
        (0): ConvBnReLU2d(
          32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], dtype=torch.uint8), observer_enabled=tensor([1], dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0, scale=tensor([1.]), zero_point=tensor([0])
            (activation_post_process): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], dtype=torch.uint8), observer_enabled=tensor([1], dtype=torch.uint8), quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.]), zero_point=tensor([0])
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (1): Identity()
        (2): Identity()
      )
      (1): ConvBn2d(
        32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False
        (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (weight_fake_quant): FakeQuantize(
          fake_quant_enabled=tensor([1], dtype=torch.uint8), observer_enabled=tensor([1], dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0, scale=tensor([1.]), zero_point=tensor([0])
          (activation_post_process): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
        )
        (activation_post_process): FakeQuantize(
          fake_quant_enabled=tensor([1], dtype=torch.uint8), observer_enabled=tensor([1], dtype=torch.uint8), quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.]), zero_point=tensor([0])
          (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
        )
      )
      (2): Identity()
    )



```python
num_train_batches = 20

# QAT takes time and one needs to train over a few epochs.
# Train and check accuracy after each epoch
for nepoch in range(8):
    train_one_epoch(qat_model, criterion, optimizer, data_loader, torch.device('cpu'), num_train_batches)
    if nepoch > 3:
        # Freeze quantizer parameters
        qat_model.apply(torch.quantization.disable_observer)
    if nepoch > 2:
        # Freeze batch norm mean and variance estimates
        qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    # Check the accuracy after each epoch
    quantized_model = torch.quantization.convert(qat_model.eval(), inplace=False)
    quantized_model.eval()
    top1, top5 = evaluate(quantized_model,criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Epoch %d :Evaluation accuracy on %d images, %2.2f'%(nepoch, num_eval_batches * eval_batch_size, top1.avg))
```

    ....................Loss tensor(0.0801, grad_fn=<DivBackward0>)
    Training: * Acc@1 97.031 Acc@5 100.000
    ........................................................................................................................................................................................................Epoch 0 :Evaluation accuracy on 10000 images, 91.78
    ....................Loss tensor(0.1211, grad_fn=<DivBackward0>)
    Training: * Acc@1 96.562 Acc@5 100.000
    ........................................................................................................................................................................................................Epoch 1 :Evaluation accuracy on 10000 images, 90.30
    ....................Loss tensor(0.1353, grad_fn=<DivBackward0>)
    Training: * Acc@1 96.094 Acc@5 100.000
    ........................................................................................................................................................................................................Epoch 2 :Evaluation accuracy on 10000 images, 89.76
    ....................Loss tensor(0.2350, grad_fn=<DivBackward0>)
    Training: * Acc@1 92.812 Acc@5 99.844
    ........................................................................................................................................................................................................Epoch 3 :Evaluation accuracy on 10000 images, 89.31
    ....................Loss tensor(0.1580, grad_fn=<DivBackward0>)
    Training: * Acc@1 94.844 Acc@5 99.844
    ........................................................................................................................................................................................................Epoch 4 :Evaluation accuracy on 10000 images, 90.72
    ....................Loss tensor(0.1394, grad_fn=<DivBackward0>)
    Training: * Acc@1 95.469 Acc@5 100.000
    ........................................................................................................................................................................................................Epoch 5 :Evaluation accuracy on 10000 images, 91.10
    ....................Loss tensor(0.1153, grad_fn=<DivBackward0>)
    Training: * Acc@1 95.469 Acc@5 100.000
    ........................................................................................................................................................................................................Epoch 6 :Evaluation accuracy on 10000 images, 91.37
    ....................Loss tensor(0.0730, grad_fn=<DivBackward0>)
    Training: * Acc@1 97.812 Acc@5 100.000
    ........................................................................................................................................................................................................Epoch 7 :Evaluation accuracy on 10000 images, 91.29



```python
torch.set_num_threads(1)
print(torch.__config__.parallel_info())
```

    ATen/Parallel:
    	at::get_num_threads() : 1
    	at::get_num_interop_threads() : 44
    OpenMP 201511 (a.k.a. OpenMP 4.5)
    	omp_get_max_threads() : 1
    Intel(R) Math Kernel Library Version 2020.0.0 Product Build 20191122 for Intel(R) 64 architecture applications
    	mkl_get_max_threads() : 1
    Intel(R) MKL-DNN v1.7.0 (Git Hash 7aed236906b1f7a05c0917e5257a1af05e9ff683)
    std::thread::hardware_concurrency() : 88
    Environment variables:
    	OMP_NUM_THREADS : [not set]
    	MKL_NUM_THREADS : [not set]
    ATen parallel backend: OpenMP
    



```python
def run_benchmark(model_file, img_loader):
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed

run_benchmark(saved_model_dir + scripted_float_model_file, data_loader_test)

run_benchmark(saved_model_dir + scripted_quantized_model_file, data_loader_test)

```

    Elapsed time:  51 ms
    Elapsed time: 107 ms





    26.741507053375244




```python
import torch.autograd.profiler as profiler
```


```python
inputs = torch.randn(5, 3, 224, 224)
```


```python
with profiler.profile(record_shapes=True) as prof:
    with profiler.record_function("model_inference"):
        model=torch.jit.load(saved_model_dir + scripted_float_model_file)
        model(inputs)
```


```python
print(prof.key_averages().table(sort_by="cpu_time_total"))
```

    ------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                              Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
    ------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                   model_inference        74.04%     350.883ms        99.99%     473.885ms     473.885ms             1  
                           forward         0.29%       1.360ms        25.76%     122.080ms     122.080ms             1  
                      aten::conv2d         0.08%     392.741us        20.37%      96.558ms       1.857ms            52  
                 aten::convolution         0.06%     297.117us        20.29%      96.165ms       1.849ms            52  
                aten::_convolution         0.11%     503.328us        20.23%      95.868ms       1.844ms            52  
        aten::_convolution_nogroup         0.05%     237.275us        11.64%      55.169ms       1.623ms            34  
                 aten::thnn_conv2d         0.03%     138.802us        11.58%      54.886ms       1.614ms            34  
         aten::thnn_conv2d_forward         0.80%       3.810ms        11.55%      54.748ms       1.610ms            34  
          aten::mkldnn_convolution         8.45%      40.061ms         8.48%      40.195ms       2.233ms            18  
                      aten::addmm_         6.51%      30.835ms         6.51%      30.835ms     181.385us           170  
                        aten::relu         0.12%     572.580us         4.81%      22.785ms     651.000us            35  
                   aten::threshold         4.65%      22.019ms         4.69%      22.212ms     634.641us            35  
                       aten::copy_         3.16%      14.965ms         3.16%      14.965ms      87.005us           172  
                      aten::select         0.37%       1.746ms         0.45%       2.117ms       4.151us           510  
                   aten::unsqueeze         0.21%     978.911us         0.28%       1.314ms       2.577us           510  
                        aten::view         0.25%       1.176ms         0.25%       1.176ms       2.034us           578  
                         aten::add         0.23%       1.082ms         0.23%       1.082ms     108.213us            10  
                  aten::as_strided         0.15%     710.509us         0.15%     710.509us       0.695us          1023  
                       aten::empty         0.13%     609.377us         0.13%     609.377us       1.643us           371  
                     aten::reshape         0.07%     349.639us         0.11%     519.225us       3.054us           170  
                        aten::set_         0.08%     391.693us         0.08%     391.693us       3.695us           106  
                          defaults         0.06%     298.378us         0.06%     298.378us       1.073us           278  
                        aten::mean         0.00%      20.213us         0.05%     224.353us     224.353us             1  
                         aten::sum         0.03%     152.363us         0.03%     164.343us     164.343us             1  
                     aten::resize_         0.02%      91.020us         0.02%      91.020us       2.677us            34  
                      aten::linear         0.00%      12.424us         0.01%      69.019us      69.019us             1  
           aten::_nnpack_available         0.01%      45.754us         0.01%      45.754us       1.346us            34  
                 aten::as_strided_         0.01%      39.987us         0.01%      39.987us       2.221us            18  
                        aten::div_         0.00%      23.015us         0.01%      39.797us      39.797us             1  
                       aten::addmm         0.01%      29.383us         0.01%      38.444us      38.444us             1  
                       aten::zeros         0.00%      13.963us         0.01%      26.042us      26.042us             1  
                      aten::detach         0.00%      21.278us         0.00%      21.278us       0.626us            34  
                           aten::t         0.00%      13.374us         0.00%      18.151us      18.151us             1  
                          aten::to         0.00%       9.276us         0.00%      16.782us      16.782us             1  
                       aten::fill_         0.00%       8.716us         0.00%       8.716us       8.716us             1  
                   aten::transpose         0.00%       2.819us         0.00%       4.777us       4.777us             1  
                      aten::expand         0.00%       3.441us         0.00%       4.282us       4.282us             1  
                       aten::zero_         0.00%       2.707us         0.00%       2.707us       2.707us             1  
               aten::empty_strided         0.00%       2.396us         0.00%       2.396us       2.396us             1  
                     aten::dropout         0.00%       2.028us         0.00%       2.028us       2.028us             1  
    ------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
    Self CPU time total: 473.911ms
    



```python
with profiler.profile(record_shapes=True) as prof:
    with profiler.record_function("model_inference"):
        model=torch.jit.load(saved_model_dir + scripted_quantized_model_file)
        model(inputs)
print(prof.key_averages().table(sort_by="cpu_time_total"))
```

    ---------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
    ---------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                  model_inference        94.72%       12.005s       100.00%       12.674s       12.674s             1  
                                     __setstate__         0.01%       1.284ms         4.78%     605.667ms      11.428ms            53  
                                  set_weight_bias         0.00%     528.731us         4.77%     604.383ms      11.403ms            53  
                        quantized::conv2d_prepack         2.11%     267.058ms         4.76%     603.508ms      11.606ms            52  
                                     aten::select         0.93%     118.037ms         1.10%     138.807ms       4.067us         34132  
                                       aten::item         0.35%      44.735ms         0.78%      98.800ms       2.894us         34134  
                  aten::q_per_channel_zero_points         0.54%      68.162ms         0.54%      68.162ms       3.994us         17066  
                                          forward         0.01%       1.498ms         0.49%      61.740ms      61.740ms             1  
                        aten::_local_scalar_dense         0.43%      54.066ms         0.43%      54.066ms       1.584us         34134  
                           quantized::conv2d_relu         0.34%      42.728ms         0.36%      46.100ms       1.317ms            35  
                       aten::q_per_channel_scales         0.24%      30.590ms         0.24%      30.590ms       1.792us         17066  
                                 aten::as_strided         0.16%      20.773ms         0.16%      20.773ms       0.609us         34133  
                                quantized::conv2d         0.10%      12.137ms         0.10%      12.283ms     722.540us            17  
                                 aten::contiguous         0.00%      10.874us         0.02%       3.036ms       3.036ms             1  
                                      aten::copy_         0.02%       2.996ms         0.02%       2.999ms       1.500ms             2  
                                       aten::set_         0.01%       1.083ms         0.01%       1.083ms       5.062us           214  
                                   quantized::add         0.00%     607.459us         0.01%     803.099us      80.310us            10  
        aten::_empty_per_channel_affine_quantized         0.00%     494.265us         0.00%     559.952us      10.565us            53  
                        aten::quantize_per_tensor         0.00%     537.419us         0.00%     537.419us     268.710us             2  
                                      aten::empty         0.00%     404.092us         0.00%     404.092us       1.063us           380  
                    aten::_empty_affine_quantized         0.00%     403.039us         0.00%     403.039us       3.445us           117  
                                       aten::mean         0.00%      28.060us         0.00%     389.534us     389.534us             1  
                        quantized::linear_prepack         0.00%     158.284us         0.00%     346.051us     346.051us             1  
                                    aten::qscheme         0.00%     235.213us         0.00%     235.213us       2.735us            86  
                                    aten::q_scale         0.00%     180.224us         0.00%     180.224us       2.120us            85  
                               aten::q_zero_point         0.00%     179.678us         0.00%     179.678us       2.114us            85  
                                 aten::dequantize         0.00%     171.384us         0.00%     177.774us      88.887us             2  
                                        aten::sum         0.00%     122.139us         0.00%     137.496us     137.496us             1  
                         aten::q_per_channel_axis         0.00%     130.843us         0.00%     130.843us       2.516us            52  
                                quantized::linear         0.00%      96.541us         0.00%     103.087us     103.087us             1  
                                         aten::to         0.00%      76.749us         0.00%      90.358us       0.844us           107  
                                       aten::div_         0.00%      22.067us         0.00%      46.738us      46.738us             1  
                                 aten::empty_like         0.00%      19.009us         0.00%      35.702us      35.702us             1  
                                      aten::zeros         0.00%      21.984us         0.00%      33.663us      33.663us             1  
                                         defaults         0.00%      13.397us         0.00%      13.397us       3.349us             4  
                                      aten::fill_         0.00%      10.314us         0.00%      10.314us      10.314us             1  
                              aten::empty_strided         0.00%       3.569us         0.00%       3.569us       3.569us             1  
                                    aten::dropout         0.00%       2.461us         0.00%       2.461us       2.461us             1  
                                      aten::zero_         0.00%       2.279us         0.00%       2.279us       2.279us             1  
    ---------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
    Self CPU time total: 12.674s
    



```python
!ls
```

    mobilenet_cifar10.pt			       mobilenetv2_cifar10.pth
    mobilenet_quantization_scripted.pth	       quant.ipynb
    mobilenet_quantization_scripted_quantized.pth

