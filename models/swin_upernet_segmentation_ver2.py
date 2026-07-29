import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .swin_transformer import SwinTransformer
from .vision_transformer import VisionTransformer
# from .simmim import SwinTransformerForSimMIM
# from .simmim import VisionTransformerForSimMIM
from .models_seg_upnet import UPerNet


class SwinEncoderForUPerNet(SwinTransformer):
    """
    Custom Swin encoder class for UPerNet decoder.
    Inherits from SwinTransformerForSimMIM and implements forward_features().
    Assumes square input images.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.depths = kwargs.get("depths")
        # self.embed_dim = kwargs.get("embed_dim")
        # assert self.num_classes == 0        

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            feat = x.transpose(1, 2).view(B, C, H, W)
            features.append(feat)
        return features  # list of 4 feature maps

class SwinUPerNet(nn.Module):
    def __init__(self, encoder, encoder_stride, num_classes):
        super().__init__()
        # Treat encoder as a direct component
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.in_chans = encoder.in_chans if hasattr(encoder, 'in_chans') else 3
        self.patch_size = encoder.patch_size if hasattr(encoder, 'patch_size') else 4

        # Derive channel dimensions for each stage
        self.in_channels_list = [256, 512, 1024, 1024]  # matches forward_features output

        # Segmentation decoder
        self.decoder = UPerNet(
            in_channels_list=self.in_channels_list,
            num_classes=num_classes
        )

    def forward(self, x):
        # Call encoder directly, like SimMIM
        # It should return a list of feature maps
        features = self.encoder.forward_features(x)  # [C1, C2, C3, C4]
        x = self.decoder(features)
        return x    
    


def build_segmentation_model(config, num_classes):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        encoder = SwinEncoderForUPerNet(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        encoder_stride = 32

    elif model_type == 'vit':
        encoder = VisionTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.VIT.PATCH_SIZE,
            in_chans=config.MODEL.VIT.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.VIT.EMBED_DIM,
            depth=config.MODEL.VIT.DEPTH,
            num_heads=config.MODEL.VIT.NUM_HEADS,
            mlp_ratio=config.MODEL.VIT.MLP_RATIO,
            qkv_bias=config.MODEL.VIT.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=config.MODEL.VIT.INIT_VALUES,
            use_abs_pos_emb=config.MODEL.VIT.USE_APE,
            use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
            use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
            use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING)
        encoder_stride = 16

    else:
        raise NotImplementedError(f"Unknown encoder type: {model_type}")
    
    model = SwinUPerNet(
        encoder=encoder,
        encoder_stride=encoder_stride,
        num_classes=num_classes
    )
    return model
