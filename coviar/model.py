"""Model definition."""

from torch import nn
from layer3_fusion import Layer3Fusion
from transforms import GroupMultiScaleCrop
from transforms import GroupRandomHorizontalFlip
import torchvision

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Model(nn.Module):
    def __init__(self, num_class, num_segments, representation, 
                 base_model='resnet152',fusion_type='add'):
        super(Model, self).__init__()
        self._representation = representation
        self.num_segments = num_segments
        self.fusion_type = fusion_type

        print(("""
Initializing model:
    base model:         {}.
    input_representation:     {}.
    num_class:          {}.
    num_segments:       {}.
        """.format(base_model, self._representation, num_class, self.num_segments)))

        self._prepare_base_model(base_model)
        self._prepare_tsn(num_class)

    def _prepare_tsn(self, num_class):

        feature_dim = getattr(self.base_model, 'fc').in_features
        self.layer3_fusion = Layer3Fusion(
            fusion_type=self.fusion_type
        )
        setattr(self.base_model, 'fc', nn.Linear(feature_dim, num_class))

        if self._representation == 'mv':
            setattr(self.base_model, 'conv1',
                    nn.Conv2d(2, 64, 
                              kernel_size=(7, 7),
                              stride=(2, 2),
                              padding=(3, 3),
                              bias=False))
            self.data_bn = nn.BatchNorm2d(2)
        if self._representation == 'residual':
            self.data_bn = nn.BatchNorm2d(3)


    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(pretrained=True)

            self._input_size = 224
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def forward(self, input):
        layer3_feature = self.extract_layer3(input)
        return self.forward_from_layer3(layer3_feature)

    def extract_layer3(self, input):

        input = input.view((-1,) + input.size()[-3:])

        if self._representation in ['mv', 'residual']:
            input = self.data_bn(input)

        x = self.base_model.conv1(input)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        layer3_feature = self.base_model.layer3(x)

        return layer3_feature

    def forward_from_layer3(self, layer3_feature):

        x = self.base_model.layer4(layer3_feature)
        x = self.base_model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.base_model.fc(x)

        return x

    @property
    def crop_size(self):
        return self._input_size

    @property
    def scale_size(self):
        return self._input_size * 256 // 224

    def get_augmentation(self):
        if self._representation in ['mv', 'residual']:
            scales = [1, .875, .75]
        else:
            scales = [1, .875, .75, .66]

        print('Augmentation scales:', scales)
        return torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, scales),
             GroupRandomHorizontalFlip(is_mv=(self._representation == 'mv'))])
