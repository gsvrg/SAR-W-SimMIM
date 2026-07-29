# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

from .simmim import build_simmim
# from .swin_upernet_segmentation import build_segmentation_model
from .swin_upernet_segmentation_ver2 import build_segmentation_model


def build_model(config, is_pretrain=True):
    if is_pretrain:
        model = build_simmim(config)
    else:
        model = build_segmentation_model(config, num_classes=config.MODEL.NUM_CLASSES)

    return model


# def build_model(config, is_pretrain=True):
#     if is_pretrain:
#         model = build_simmim(config)
#     else:
#         model_type = config.MODEL.TYPE
#         if model_type == 'swin':
#             model = build_swin(config)
#         elif model_type == 'vit':
#             model = build_vit(config)
#         else:
#             raise NotImplementedError(f"Unknown fine-tune model: {model_type}")

#     return model