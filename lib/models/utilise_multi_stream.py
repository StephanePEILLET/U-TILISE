import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from lib.models.fusion_blocks import GatedSEFusion
from lib.models.ltae_transformer import LTAEtransformer
from lib.models.parameters import NormType, TemporalAggregationMode, UpConvType
from lib.models.utilise import (
    ConvBlock,
    DownConvBlock,
    TemporalAggregator,
    UpConvBlock,
    str2ActivationType,
)


class SkipFusionWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, sar_asc, sar_desc, opt_hr, opt_lr):
        return self.module(opt_hr, opt_lr, sar_asc, sar_desc)


class UtiliseMultiStream(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            use_sar: str = "both",
            max_attention_size: int = 32,
            encoder_widths=[64, 64, 64, 128],
            decoder_widths=[32, 32, 64, 128],
            use_gradient_checkpointing: bool = False,
            **kwargs):
        super().__init__()

        self.use_sar = use_sar
        self.max_attention_size = max_attention_size
        self.use_gradient_checkpointing = use_gradient_checkpointing

        k, s, p = 3, 1, 1
        str_conv_k, str_conv_s, str_conv_p = 4, 2, 1
        str_conv_k_up, str_conv_p_up = 2, 0
        pad_value = None
        norm = NormType.BATCH

        self.opt_hr_in = 10
        self.opt_lr_in = 0
        self.sar_in = 4 if self.use_sar in ['asc', 'desc', 'both'] else 0
        self.use_opt_lr = self.opt_lr_in > 0
        activation = str2ActivationType('relu')

        # Encoders
        self.in_conv_opt_hr = ConvBlock(n_kernels=[self.opt_hr_in, encoder_widths[0], encoder_widths[0]], norm=norm, activation=activation, pad_value=pad_value, padding_mode='zeros')
        if self.use_sar in ['asc', 'both']:
            self.in_conv_sar_asc = ConvBlock(n_kernels=[self.sar_in, encoder_widths[0], encoder_widths[0]], norm=norm, activation=activation, pad_value=pad_value, padding_mode='zeros')
        if self.use_sar in ['desc', 'both']:
            self.in_conv_sar_desc = ConvBlock(n_kernels=[self.sar_in, encoder_widths[0], encoder_widths[0]], norm=norm, activation=activation, pad_value=pad_value, padding_mode='zeros')

        self.down_blocks_opt_hr = nn.ModuleList()
        self.down_blocks_sar_asc = nn.ModuleList() if self.use_sar in ['asc', 'both'] else None
        self.down_blocks_sar_desc = nn.ModuleList() if self.use_sar in ['desc', 'both'] else None

        for i in range(len(encoder_widths) - 1):
            d_in = encoder_widths[i]
            d_out = encoder_widths[i + 1]
            self.down_blocks_opt_hr.append(DownConvBlock(d_in=d_in, d_out=d_out, k=str_conv_k, s=str_conv_s, p=str_conv_p, pad_value=pad_value, norm=norm, activation=activation, padding_mode='zeros'))
            if self.use_sar in ['asc', 'both']: self.down_blocks_sar_asc.append(DownConvBlock(d_in=d_in, d_out=d_out, k=str_conv_k, s=str_conv_s, p=str_conv_p, pad_value=pad_value, norm=norm, activation=activation, padding_mode='zeros'))
            if self.use_sar in ['desc', 'both']: self.down_blocks_sar_desc.append(DownConvBlock(d_in=d_in, d_out=d_out, k=str_conv_k, s=str_conv_s, p=str_conv_p, pad_value=pad_value, norm=norm, activation=activation, padding_mode='zeros'))

        # Bottleneck Temporal Encoding (LTAE) & Aggregator
        self.ltae_opt = LTAEtransformer(in_channels=encoder_widths[-1], return_att=True)
        self.agg_opt = TemporalAggregator(mode=TemporalAggregationMode("att_group"))
        if self.use_sar in ['asc', 'both']:
            self.ltae_sar_asc = LTAEtransformer(in_channels=encoder_widths[-1], return_att=True)
            self.agg_sar_asc = TemporalAggregator(mode=TemporalAggregationMode("att_group"))
        if self.use_sar in ['desc', 'both']:
            self.ltae_sar_desc = LTAEtransformer(in_channels=encoder_widths[-1], return_att=True)
            self.agg_sar_desc = TemporalAggregator(mode=TemporalAggregationMode("att_group"))

        # Bottleneck Fusion (SE/Gate)
        bottleneck_dim = encoder_widths[-1]
        self.bottleneck_fusion = GatedSEFusion(bottleneck_dim, 0, bottleneck_dim if self.use_sar in ['asc', 'both'] else 0, bottleneck_dim if self.use_sar in ['desc', 'both'] else 0)

        # Cross Modal Skip Connections - All using GatedSEFusion for memory efficiency
        self.skip_fusions = nn.ModuleList()
        for i in range(len(encoder_widths)):
            dim = encoder_widths[i]
            sar_d = dim if self.use_sar in ['asc', 'both'] else 0
            # Using GatedSEFusion everywhere for memory optimization
            self.skip_fusions.append(SkipFusionWrapper(GatedSEFusion(dim, 0, sar_d, sar_d)))

        self.up_blocks = nn.ModuleList()
        for i in range(len(encoder_widths) - 1):
            idx = (len(encoder_widths) - 1) - i  # 3, 2, 1 if len is 4
            d_in = decoder_widths[idx]
            d_out = decoder_widths[idx - 1]
            d_skip = encoder_widths[idx - 1]

            self.up_blocks.append(
                UpConvBlock(
                    d_in=d_in,
                    d_out=d_out,
                    d_skip=d_skip,
                    k=str_conv_k_up,
                    s=str_conv_s,
                    p=str_conv_p_up,
                    upconv_type=UpConvType.TRANSPOSE,
                    pad_value=pad_value,
                    norm_conv=norm, norm_skip=norm,
                    activation=activation,
                    padding_mode='zeros'
                )
            )

        self.out_conv = ConvBlock(
            n_kernels=[decoder_widths[0], output_dim],
            pad_value=pad_value,
            norm=NormType.NONE,
            activation=None,
            k=1,
            p=0,
            padding_mode='zeros'
        )

    def forward(self, input, batch=None, batch_positions=None, **kwargs):
        # Determine the device from input or model parameters
        device = next(self.parameters()).device

        if type(input) is dict and 'x_dict' in input:
            x_dict = input['x_dict']
            cld = input.get('cloud_mask')
        elif batch is not None and 'x_dict' in batch:
            x_dict = batch['x_dict']
            cld = batch.get('cloud_mask')
        else:
            x_dict = {"opt_hr": input[:, :, :10]}
            cld = torch.zeros(input.shape[0], input.shape[1], 1, input.shape[3], input.shape[4]).to(device)

        # Ensure all tensors are on the correct device
        x_opt_hr = x_dict.get("opt_hr")
        if x_opt_hr is not None:
            x_opt_hr = x_opt_hr.to(device)
        x_sar_asc = x_dict.get("sar_asc")
        if x_sar_asc is not None:
            x_sar_asc = x_sar_asc.to(device)
        x_sar_desc = x_dict.get("sar_desc")
        if x_sar_desc is not None:
            x_sar_desc = x_sar_desc.to(device)

        B, T, C, H, W = x_opt_hr.shape

        if cld is None:
            cld = torch.zeros(B, T, 1, H, W).to(device)
        else:
            cld = cld.to(device)

        # Optional pad mask logic
        pad_mask = (x_opt_hr == 0).all(dim=-1).all(dim=-1).all(dim=-1)  # simple fallback
        if batch_positions is not None:
            batch_positions_device = batch_positions.to(x_opt_hr.device)
            pad_mask = torch.logical_and(pad_mask, batch_positions_device == 0)

        # Encoding with optional gradient checkpointing
        if self.use_gradient_checkpointing and self.training:
            h_opt_hr = checkpoint(
                self.in_conv_opt_hr.smart_forward, x_opt_hr, pad_mask,
                use_reentrant=False
            )
            h_sar_asc = checkpoint(
                self.in_conv_sar_asc.smart_forward, x_sar_asc, pad_mask,
                use_reentrant=False
            ) if x_sar_asc is not None else None
            h_sar_desc = checkpoint(
                self.in_conv_sar_desc.smart_forward, x_sar_desc, pad_mask,
                use_reentrant=False
            ) if x_sar_desc is not None else None
        else:
            h_opt_hr = self.in_conv_opt_hr.smart_forward(x_opt_hr, pad_mask=pad_mask)
            h_sar_asc = self.in_conv_sar_asc.smart_forward(x_sar_asc, pad_mask=pad_mask) if x_sar_asc is not None else None
            h_sar_desc = self.in_conv_sar_desc.smart_forward(x_sar_desc, pad_mask=pad_mask) if x_sar_desc is not None else None

        saved_features = [{"opt_hr": h_opt_hr, "sar_asc": h_sar_asc, "sar_desc": h_sar_desc}]

        for i in range(len(self.down_blocks_opt_hr)):
            if self.use_gradient_checkpointing and self.training:
                h_opt_hr = checkpoint(
                    self.down_blocks_opt_hr[i].smart_forward, h_opt_hr, pad_mask,
                    use_reentrant=False
                )
                if self.down_blocks_sar_asc is not None and h_sar_asc is not None:
                    h_sar_asc = checkpoint(
                        self.down_blocks_sar_asc[i].smart_forward, h_sar_asc, pad_mask,
                        use_reentrant=False
                    )
                if self.down_blocks_sar_desc is not None and h_sar_desc is not None:
                    h_sar_desc = checkpoint(
                        self.down_blocks_sar_desc[i].smart_forward, h_sar_desc, pad_mask,
                        use_reentrant=False
                    )
            else:
                h_opt_hr = self.down_blocks_opt_hr[i].smart_forward(h_opt_hr, pad_mask=pad_mask)
                if self.down_blocks_sar_asc is not None and h_sar_asc is not None:
                    h_sar_asc = self.down_blocks_sar_asc[i].smart_forward(h_sar_asc, pad_mask=pad_mask)
                if self.down_blocks_sar_desc is not None and h_sar_desc is not None:
                    h_sar_desc = self.down_blocks_sar_desc[i].smart_forward(h_sar_desc, pad_mask=pad_mask)
            saved_features.append({"opt_hr": h_opt_hr, "sar_asc": h_sar_asc, "sar_desc": h_sar_desc})

        # Temporal LTAE Bottleneck
        ltae_o_out, ltae_o_att = self.ltae_opt(h_opt_hr, batch_positions=batch_positions, pad_mask=pad_mask)
        bn_opt = self.agg_opt(ltae_o_out, pad_mask=pad_mask, attn_mask=ltae_o_att)

        bn_sar_asc, bn_sar_desc = None, None
        if h_sar_asc is not None:
            l_out, l_att = self.ltae_sar_asc(h_sar_asc, batch_positions=batch_positions, pad_mask=pad_mask)
            bn_sar_asc = self.agg_sar_asc(l_out, pad_mask=pad_mask, attn_mask=l_att)
        if h_sar_desc is not None:
            l_out, l_att = self.ltae_sar_desc(h_sar_desc, batch_positions=batch_positions, pad_mask=pad_mask)
            bn_sar_desc = self.agg_sar_desc(l_out, pad_mask=pad_mask, attn_mask=l_att)

        gate_mask_bottleneck = 1.0 - cld

        # We need to flatten B, T into B*T for Spatial Fusion module
        # bn_opt is B x T x C x H x W

        bn_opt = bn_opt.view(-1, *bn_opt.shape[2:])
        bn_sar_asc = bn_sar_asc.view(-1, *bn_sar_asc.shape[2:]) if bn_sar_asc is not None else None
        bn_sar_desc = bn_sar_desc.view(-1, *bn_sar_desc.shape[2:]) if bn_sar_desc is not None else None
        gate_mask_flat = gate_mask_bottleneck.view(-1, *gate_mask_bottleneck.shape[2:])

        fused_bn = self.bottleneck_fusion(bn_opt, None, bn_sar_asc, bn_sar_desc, gate_mask_flat)
        x = fused_bn.view(B, T, *fused_bn.shape[1:])  # B x T x C x H x W

        # Decoder loops
        for i, up_block in enumerate(self.up_blocks):
            feat_dict = saved_features[-(i + 2)]

            # Flatten B*T -> pass skip -> view back
            skip_opt = feat_dict['opt_hr'].view(-1, *feat_dict['opt_hr'].shape[2:]) if feat_dict['opt_hr'] is not None else None
            skip_sar_a = feat_dict['sar_asc'].view(-1, *feat_dict['sar_asc'].shape[2:]) if feat_dict['sar_asc'] is not None else None
            skip_sar_d = feat_dict['sar_desc'].view(-1, *feat_dict['sar_desc'].shape[2:]) if feat_dict['sar_desc'] is not None else None

            fused_skip = self.skip_fusions[-(i + 2)](skip_sar_a, skip_sar_d, skip_opt, None)

            fused_skip = fused_skip.view(B, T, *fused_skip.shape[1:])

            x = up_block(x, fused_skip, pad_mask=pad_mask)

        out = self.out_conv.smart_forward(x, pad_mask=pad_mask)
        return out
