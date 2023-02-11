import torchvision

from ..models.timm_wrapper import TimmModelsWrapper
from ..models.efficient_net_pytcv import efficientnet_b0
from ..torch_utils import load_pretrained_weights


__AVAI_MODELS__ = {'resnet_18', 'mobilenetv3_large_21k', 'efficientnetv2_s', 'efficientnet_b0'}


def build_model(config):
    assert config.model.name in __AVAI_MODELS__, f"Wrong model name parameter. Expected one of {__AVAI_MODELS__}"

    if config.model.name == 'mobilenetv3_large_21k':
        model = TimmModelsWrapper('mobilenetv3_large_100_miil', pretrained=config.model.pretrained,
                                            num_classes=config.model.num_classes)
    elif config.model.name == 'resnet_18':
        weights = None if not config.model.pretrained else torchvision.models.ResNet18_Weights.DEFAULT
        model = torchvision.models.resnet18(num_classes=config.model.num_classes, weights=weights)
    elif config.model.name == 'efficientnetv2_s':
        model = TimmModelsWrapper('tf_efficientnetv2_s_in21k', pretrained=config.model.pretrained,
                                  num_classes=config.model.num_classes)
    elif config.model.name == 'efficientnet_b0':
        model = efficientnet_b0(pretrained=config.model.load_weights, num_classes=config.model.num_classes)

    if config.model.load_weights:
        load_pretrained_weights(model, config.model.load_weights)

    return model