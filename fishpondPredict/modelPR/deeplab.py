from collections import OrderedDict

from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from modelPR.resnet import resnet103, resnet53
from torchvision.models import resnet50, resnet101


class SmallDeepLab(_SimpleSegmentationModel):
    def forward(self, input_):
        result = self.backbone(input_)
        result["coarse"] = self.classifier(result["out"])
        return result


def deeplabv3(pretrained=False, resnet="res50", head_in_ch=2048, num_classes=21):
    resnet = {
        "res53":  resnet53,
        "res103": resnet103,
        "res50":  resnet50,
        "res101": resnet101
    }[resnet]

    net = SmallDeepLab(
        backbone=IntermediateLayerGetter(
            resnet(pretrained=True, replace_stride_with_dilation=[False, True, True]),
            return_layers={'layer2': 'res2', 'layer4': 'out'}
        ),
        classifier=DeepLabHead(head_in_ch, num_classes)
    )
    return net


if __name__ == "__main__":
    import torch
    x = torch.randn(3, 3, 512, 1024).cuda()
    net = deeplabv3(False).cuda()
    result = net(x)
    for k, v in result.items():
        print(k, v.shape)
