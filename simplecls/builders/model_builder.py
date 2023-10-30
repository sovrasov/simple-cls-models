import torchvision

from ..models.timm_wrapper import TimmModelsWrapper
from ..models.efficient_net_pytcv import efficientnet_b0
from ..torch_utils import load_pretrained_weights


__AVAI_MODELS__ = {'resnet_18', 'mobilenetv3_large_21k', 'efficientnetv2_s', 'efficientnet_b0', 'swin_small', 'swin_mmdet'}


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
    elif config.model.name == 'swin_small':
        model = TimmModelsWrapper('swin_tiny_patch4_window7_224.ms_in1k', pretrained=config.model.pretrained,
                                  num_classes=config.model.num_classes)
    elif config.model.name == 'swin_mmdet':
        from ..models.swin_mmdet import get_swin
        model = get_swin(config.model.num_classes, embed_dims=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            patch_norm=True,
            out_indices=(3,),
            with_cp=False,
            convert_weights=True,
            init_cfg=dict(type="Pretrained", checkpoint="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"))

    elif config.model.name == 'efficientnet_b0':
        model = efficientnet_b0(pretrained=config.model.load_weights, num_classes=config.model.num_classes)

    if config.model.load_weights:
        load_pretrained_weights(model, config.model.load_weights)

    return model