import torch
import torch.nn.functional as F
from torch import nn


class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // 4, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GatedSEFusion(nn.Module):
    def __init__(self, opt_hr_dim, opt_lr_dim=0, sar_asc_dim=0, sar_desc_dim=0):
        super().__init__()
        self.opt_lr_dim = opt_lr_dim
        self.sar_asc_dim = sar_asc_dim
        self.sar_desc_dim = sar_desc_dim
        total_dim = opt_hr_dim + opt_lr_dim + sar_asc_dim + sar_desc_dim
        self.se = SE_Block(total_dim)
        self.proj = nn.Conv2d(total_dim, opt_hr_dim, kernel_size=1)

    def forward(self, f_opt_hr, f_opt_lr=None, f_sar_asc=None, f_sar_desc=None, gate_mask=None):
        out_opt_hr = f_opt_hr

        # Apply gate on Optical only (gate_mask is probability of clear sky: 1=Clear, 0=Cloud)
        if gate_mask is not None:
            # Resize mask to current feature map resolution
            gate_mask_resized = F.interpolate(gate_mask, size=f_opt_hr.shape[-2:], mode='nearest')
            out_opt_hr = out_opt_hr * gate_mask_resized

        to_concat = [out_opt_hr]

        if self.opt_lr_dim > 0:
            if f_opt_lr is not None:
                out_opt_lr = f_opt_lr
                if gate_mask is not None:
                    out_opt_lr = out_opt_lr * gate_mask_resized
                to_concat.append(out_opt_lr)
            else:
                to_concat.append(torch.zeros_like(out_opt_hr))

        if self.sar_asc_dim > 0:
            if f_sar_asc is not None:
                to_concat.append(f_sar_asc)
            else:
                to_concat.append(torch.zeros_like(out_opt_hr))

        if self.sar_desc_dim > 0:
            if f_sar_desc is not None:
                to_concat.append(f_sar_desc)
            else:
                to_concat.append(torch.zeros_like(out_opt_hr))

        fused = torch.cat(to_concat, dim=1)  # Concat along channel dimension
        fused = self.se(fused)
        out = self.proj(fused)
        return out


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, q_input, k_input, v_input):
        B, C, H, W = q_input.shape
        # Flatten spatial dimensions
        q = self.to_q(q_input).view(B, self.num_heads, C // self.num_heads, H * W).transpose(-2, -1)
        k = self.to_k(k_input).view(B, self.num_heads, C // self.num_heads, H * W).transpose(-2, -1)
        v = self.to_v(v_input).view(B, self.num_heads, C // self.num_heads, H * W).transpose(-2, -1)

        out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(-2, -1).reshape(B, C, H, W)
        out = self.proj(out)
        return out


class SARGuidedFusion(nn.Module):
    def __init__(self, opt_hr_dim, sar_total_dim):
        super().__init__()
        # Project everything to the same dimension for Attention
        self.dim = opt_hr_dim

        self.proj_sar = nn.Conv2d(sar_total_dim, self.dim, kernel_size=1) if sar_total_dim != self.dim else nn.Identity()
        self.proj_opt = nn.Conv2d(opt_hr_dim, self.dim, kernel_size=1)

        self.attn = CrossAttentionBlock(self.dim)

        # After fetching the features with attention, recalibrate with SE
        # The input here is concatenated (Gated OptHR) + (Attention Output)
        self.se = SE_Block(self.dim * 2)
        self.out_proj = nn.Conv2d(self.dim * 2, opt_hr_dim, kernel_size=1)

    def forward(self, f_opt_hr, f_opt_lr=None, f_sar_asc=None, f_sar_desc=None, gate_mask=None):
        # 1. Gating
        out_opt_hr = f_opt_hr
        if gate_mask is not None:
            gate_mask_resized = F.interpolate(gate_mask, size=f_opt_hr.shape[-2:], mode='nearest')
            out_opt_hr = out_opt_hr * gate_mask_resized

        # 2. Prepare SAR structure (Query & Key)
        sar_list = []
        if f_sar_asc is not None: sar_list.append(f_sar_asc)
        if f_sar_desc is not None: sar_list.append(f_sar_desc)
        f_sar = torch.cat(sar_list, dim=1) if sar_list else None

        if f_sar is None:
            # Fallback if no SAR: just SE
            fused = torch.cat([out_opt_hr, out_opt_hr], dim=1)
            fused = self.se(fused)
            return self.out_proj(fused)

        sar_struct = self.proj_sar(f_sar)
        opt_values = self.proj_opt(out_opt_hr)

        # 3. Cross Attention: SAR asks Optical to fill in the color
        attn_out = self.attn(sar_struct, sar_struct, opt_values)

        # 4. Squeeze-and-Excitation / Mixing
        fused = torch.cat([out_opt_hr, attn_out], dim=1)
        fused = self.se(fused)
        out = self.out_proj(fused)

        # Residual
        return out + f_opt_hr


class HierarchicalCrossModalSkipConnection(nn.Module):
    def __init__(self, dim_sar_a, dim_sar_d, dim_opt_hr, dim_opt_lr, dim_out, num_heads=4, reduction_ratio=4):
        """
        dim_sar_a, dim_sar_d : Nombres de canaux pour SAR Ascendant et Descendant
        dim_opt_hr, dim_opt_lr : Nombres de canaux pour Optique HR (RGB/NIR) et LR (RedEdge/SWIR)
        dim_out : Dimension de sortie souhaitée pour le décodeur
        """
        super().__init__()
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.scale = self.head_dim ** -0.5

        # --- 1. PRE-FUSION INTRA-MODALE ---
        # On concatène et on utilise un Conv 1x1 pour mélanger les sous-parties
        self.fuse_sar = nn.Sequential(
            nn.Conv2d(dim_sar_a + dim_sar_d, dim_out, kernel_size=1) if dim_sar_a + dim_sar_d > 0 else nn.Conv2d(dim_out, dim_out, kernel_size=1),
            nn.BatchNorm2d(dim_out),
            nn.GELU()
        )

        self.fuse_opt = nn.Sequential(
            nn.Conv2d(dim_opt_hr + dim_opt_lr, dim_out, kernel_size=1) if dim_opt_hr + dim_opt_lr > 0 else nn.Conv2d(dim_out, dim_out, kernel_size=1),
            nn.BatchNorm2d(dim_out),
            nn.GELU()
        )

        # --- 2. PREPARATION POUR L'ATTENTION (Q, K, V) ---
        self.q_proj = nn.Conv2d(dim_out, dim_out, kernel_size=1)
        self.k_proj = nn.Conv2d(dim_out, dim_out, kernel_size=1)
        self.v_proj = nn.Conv2d(dim_out, dim_out, kernel_size=1)

        # --- 3. SPATIAL REDUCTION (SRA) POUR L'OPTIQUE ---
        self.sr_ratio = reduction_ratio
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim_out, dim_out, kernel_size=reduction_ratio, stride=reduction_ratio)
            self.norm = nn.LayerNorm(dim_out)

        # --- 4. PROJECTION FINALE ---
        self.out_proj = nn.Conv2d(dim_out, dim_out, kernel_size=1)

    def forward(self, sar_asc, sar_desc, opt_hr, opt_lr):
        B, _, H, W = opt_hr.shape if opt_hr is not None else sar_asc.shape
        device = opt_hr.device if opt_hr is not None else sar_asc.device

        sars = []
        if sar_asc is not None: sars.append(sar_asc)
        if sar_desc is not None: sars.append(sar_desc)
        dim_sars_total = (sar_asc.shape[1] if sar_asc is not None else 0) + (sar_desc.shape[1] if sar_desc is not None else 0)
        if len(sars) > 0:
            x_sar_cat = torch.cat(sars, dim=1)
        else:
            in_c = self.fuse_sar[0].in_channels
            x_sar_cat = torch.zeros(B, in_c, H, W, device=device)

        opts = []
        if opt_hr is not None: opts.append(opt_hr)
        if opt_lr is not None: opts.append(opt_lr)
        if len(opts) > 0:
            x_opt_cat = torch.cat(opts, dim=1)
        else:
            in_c_opt = self.fuse_opt[0].in_channels
            x_opt_cat = torch.zeros(B, in_c_opt, H, W, device=device)

        # --- ETAPE 1 : Pré-fusion des sous-modalités ---
        x_sar_fused = self.fuse_sar(x_sar_cat)
        x_opt_fused = self.fuse_opt(x_opt_cat)

        # --- ETAPE 2 : Le SAR pose la question (QUERY) ---
        q = self.q_proj(x_sar_fused)
        q = q.flatten(2).transpose(1, 2).reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)

        # --- ETAPE 3 : Réduction Spatiale de l'Optique (SRA) ---
        if self.sr_ratio > 1:
            x_opt_reduced = self.sr(x_opt_fused)
            x_opt_reduced = x_opt_reduced.flatten(2).transpose(1, 2)
            x_opt_reduced = self.norm(x_opt_reduced).transpose(1, 2).reshape(B, self.dim_out, H // self.sr_ratio, W // self.sr_ratio)
        else:
            x_opt_reduced = x_opt_fused

        # --- ETAPE 4 : L'Optique fournit la réponse (KEY & VALUE) ---
        k = self.k_proj(x_opt_reduced)
        v = self.v_proj(x_opt_reduced)

        k = k.flatten(2).transpose(1, 2).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # --- ETAPE 5 : La Cross-Attention (Flash Attention) ---
        x_cross = F.scaled_dot_product_attention(q, k, v)
        x_cross = x_cross.transpose(1, 2).reshape(B, H, W, self.dim_out).permute(0, 3, 1, 2)

        # Projection finale et connexion résiduelle (on ajoute le SAR global de base)
        out = self.out_proj(x_cross) + x_sar_fused

        return out
