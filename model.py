import sys

import torch.nn as nn
import torch
import timm

import numpy as np
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import  repeat
import math
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
from timm.layers import get_norm_layer
tensor_rotate = partial(rotate,interpolation=InterpolationMode.BILINEAR)


class NormMlpClassifierHead(nn.Module):
    """ A Pool -> Norm -> Mlp Classifier Head for '2D' NCHW tensors
    """
    def __init__(
            self,
            in_features: int,
            pool_type: str = 'avg',
            drop_rate: float = 0.,
    ):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        norm_layer = get_norm_layer('layernorm2d')
        self.norm = norm_layer(in_features)
        self.flatten = nn.Flatten(1) if pool_type else nn.Identity()
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.drop(x)
        if pre_logits:
            return x
        return x

    
class CA(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.GELU() #nn.PReLU(in_planes // ratio)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out
        return self.sigmoid(out)


class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(max_out)
        return self.sigmoid(x)
    
class BlancedAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        """
        Title: BAM: A Balanced Attention Mechanism for Single Image Super Resolution
        paper: https://arxiv.org/abs/2104.07566
        code: https://github.com/dandingbudanding/BAM
        """
        super(BlancedAttention, self).__init__()
        self.ca = CA(in_planes, in_planes // reduction)
        self.sa = SA()
        # self.conv=nn.Conv2d(in_planes*2,in_planes,1)
        # self.norm=nn.BatchNorm2d(in_planes)

    def forward(self, x):
        ca_ch = self.ca(x)
        sa_ch = self.sa(x)
        out = ca_ch.mul(sa_ch) * x
        # out_fused = self.conv(torch.cat([ca_ch, sa_ch], dim=1))
        return out
    

class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops
    

class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b c h w
        return x

    def flops(self):
        flops = 0
        return flops
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MSIE(nn.Module):
    def __init__(self, in_ch, out_ch=16, strip=5, stride=1) -> None:
        super(MSIE, self).__init__()
        self.act = nn.GELU()

        a0_out_feat = out_ch * 4
        self.ou_dim = out_ch * 4 + a0_out_feat
        self.conv_a0 = nn.Sequential(nn.Conv2d(in_ch, a0_out_feat, 3, 1, padding=1,dilation=1, bias=False),nn.BatchNorm2d(a0_out_feat),self.act)

        self.multi_conv1 = nn.Conv2d(in_ch, out_ch, (1, strip), stride=stride, padding=(0, strip//2))
        self.multi_conv2 = nn.Conv2d(in_ch, out_ch, (strip, 1), stride=stride, padding=(strip//2, 0))
        self.multi_conv3 = nn.Conv2d(in_ch, out_ch, (1, strip), stride=stride, padding=(0, strip//2))
        self.multi_conv4 = nn.Conv2d(in_ch, out_ch, (1, strip), stride=stride, padding=(0, strip//2))
        
        self.att = BlancedAttention(self.ou_dim, reduction=8)
        self.angle = [0,45,90,135,180]
    
    def forward(self, x):
        x0 = self.multi_conv1(x)                                        # 0°
        x90 = self.multi_conv2(x)                                       # 90°

        x45r = self.multi_conv3(tensor_rotate(x, self.angle[1]))        # 45° in rotated frame
        x135r = self.multi_conv4(tensor_rotate(x, self.angle[3]))       # 135° in rotated frame
        x45 = tensor_rotate(x45r, -self.angle[1])                        # back to original
        x135 = tensor_rotate(x135r, -self.angle[3])                      # back

        x_a0 = self.conv_a0(x)
        feat = torch.cat((x0, x90, x45, x135), dim=1)
        feat = torch.cat([feat, x_a0], dim=1)
        feat = self.att(feat)                                           # [B, out_ch*4 + a0_out_feat, H, W]
        
        M0   = x0.mean(dim=1, keepdim=False)
        M90  = x90.mean(dim=1, keepdim=False)
        M45  = x45.mean(dim=1, keepdim=False)
        M135 = x135.mean(dim=1, keepdim=False)
        M_dir = torch.stack([M0, M45, M90, M135], dim=1)                # [B,4,H,W]   [0°,45°,90°,135°]

        return feat, M_dir
    

class MSIEProj(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(4, eps=eps)

    def forward(self, M):  # M: [B,4,H,W] → K: [B,4,H,W]
        B, _, H, W = M.shape
        ph0   = M[:, 0].mean(dim=1, keepdim=True)        # [B,1,W]
        dh    = 0.5 * (M[:, 1].mean(dim=1, keepdim=True) # [B,1,W]
                     + M[:, 3].mean(dim=1, keepdim=True))# [B,1,W]
        ph    = (ph0 + dh).unsqueeze(2).expand(B, 1, H, W)  # [B,1,H,W]

        pv90  = M[:, 2].mean(dim=2, keepdim=True)           # [B,H,1]
        dv    = 0.5 * (M[:, 1].mean(dim=2, keepdim=True)    # [B,H,1]
                     + M[:, 3].mean(dim=2, keepdim=True))   # [B,H,1]
        pv    = (pv90 + dv).unsqueeze(1).expand(B, 1, H, W) # [B,1,H,W]

        def norm01(t):
            tmin = t.amin(dim=(2,3), keepdim=True)
            tmax = t.amax(dim=(2,3), keepdim=True)
            return (t - tmin) / (tmax - tmin + self.eps)

        K = torch.cat([ph, pv, ph, pv], dim=1)  # [B,4,H,W]
        K = self.ln(K.permute(0,2,3,1)).permute(0,3,1,2)  # [B,4,H,W]
        return K


class MSIEProj4Dirs(nn.Module):
    def __init__(self, use_layernorm: bool = True, eps: float = 1e-6):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.eps = eps
        if use_layernorm:
            self.ln = nn.LayerNorm(4, eps=eps)

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        """
        ch0=0°(L→R), ch1=90°(T→B), ch2=45°(R→L), ch3=135°(B→T)
        """
        B, C, H, W = M.shape
        assert C == 4, "Expect M to have 4 direction channels: [0°,45°,90°,135°]"
        ph0 = M[:, 0].mean(dim=1, keepdim=True)         # [B,1,W]
        dh  = 0.5*(M[:, 1].mean(dim=1, keepdim=True) +
                   M[:, 3].mean(dim=1, keepdim=True))   # [B,1,W]
        ph  = (ph0 + dh).unsqueeze(2).expand(B, 1, H, W)  # [B,1,H,W]
        pv90 = M[:, 2].mean(dim=2, keepdim=True)        # [B,H,1]
        dv   = 0.5*(M[:, 1].mean(dim=2, keepdim=True) +
                    M[:, 3].mean(dim=2, keepdim=True))  # [B,H,1]
        pv   = (pv90 + dv).unsqueeze(1).expand(B, 1, H, W) # [B,1,H,W]

        def norm01(t):
            tmin = t.amin(dim=(2,3), keepdim=True)
            tmax = t.amax(dim=(2,3), keepdim=True)
            return (t - tmin) / (tmax - tmin + self.eps)

        ph = norm01(ph)
        pv = norm01(pv)

        ch0 = ph                         # 0°: L→R
        ch1 = pv                         # 90°: T→B
        ch2 = torch.flip(ph, dims=[3])   # 45°: R→L  
        ch3 = torch.flip(pv, dims=[2])   # 135°: B→T 

        K = torch.cat([ch0, ch1, ch2, ch3], dim=1)  # [B,4,H,W]

        if self.use_layernorm:
            K = self.ln(K.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B,4,H,W]
        return K

class HyperHead(nn.Module):
    def __init__(self, d_state: int, hidden: int = 128):
        super().__init__()
        self.d_state = d_state
        self.mlp = nn.Linear(4, 4 * d_state)

    def forward(self, M_dir: torch.Tensor):
        r = M_dir.mean(dim=(2,3))          # [B,4]
        s = self.mlp(r)                    # [B, 4*d_state]
        s = torch.tanh(s.view(-1, 4, self.d_state))
        s = s - s.mean(dim=(1,2), keepdim=True)
        # s = torch.sigmoid(s.view(-1, 4, self.d_state))
        return s
    
class SS2D(nn.Module):
    def __init__(
            self, d_model, d_state=16, d_conv=3, expand=2., dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", 
            dt_scale=1.0, dt_init_floor=1e-4, dropout=0., conv_bias=True, bias=False, device=None, dtype=None,
            #======
            k_sem: int = 10,               
            use_c_mod: bool = False,          
            a_mode: str = "lite",              
            shareA: bool = True,          
            alpha_A_max: float = 0.20,      
            gamma_B_max: float = 0.06,     
            gamma_C_scale: float = 0.5,      
            use_entropy_for_B: bool = True,   
            is_mod_BC: bool = True,
            **kwargs,):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(in_channels=self.d_inner,out_channels=self.d_inner,groups=self.d_inner,bias=conv_bias,kernel_size=d_conv,padding=(d_conv - 1) // 2,**factory_kwargs,)
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.is_mod_BC = is_mod_BC
        if self.is_mod_BC:
            self.hyper_B = HyperHead(d_state=self.d_state, hidden=128)
            self.use_c_mod = use_c_mod
            if self.use_c_mod:
                self.hyper_C = HyperHead(d_state=self.d_state, hidden=128)

        self.gamma_B_max = gamma_B_max
        self.gamma_C_scale = gamma_C_scale
        self.use_entropy_for_B = use_entropy_for_B

        self.alpha_A_max = alpha_A_max
        self.a_mode = a_mode
        self.shareA = shareA
        self.k_sem = k_sem

        self.msieProjDirs = MSIEProj4Dirs()
        self.msieProj = MSIEProj()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        # print('A', A.shape)
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    
    def _entropy_gamma(self, M_dir: torch.Tensor):
        B = M_dir.size(0)
        if not self.use_entropy_for_B:
            return torch.full((B,1,1,1), self.gamma_B_max, device=M_dir.device, dtype=M_dir.dtype)
        
        r = M_dir.mean(dim=(2,3))                      # [B,4]
        g_dir = torch.softmax(r, dim=1)                # [B,4]
        Hdir = -(g_dir * (g_dir.clamp_min(1e-8)).log()).sum(dim=1)  # [B]
        q = 1.0 - Hdir / math.log(4.0)                 # [B]
        gamma = (self.gamma_B_max * q).view(B, 1, 1, 1)
        return gamma

    def _modulate_BC(self, Bs, Cs, M_dir, use_c: bool):
        Bsz, K, Dst, L = Bs.shape
        M_dir = self.msieProjDirs(M_dir)
        scale_B = self.hyper_B(M_dir).unsqueeze(-1)      # [B,4,d_state,1]
        gamma_B = self._entropy_gamma(M_dir)  # [B,1,1,1]
        Bs_mod = Bs * (1.0 + gamma_B * scale_B)  
        # Bs_mod = (1.0-gamma_B)*Bs + gamma_B * Bs * scale_B

        # K_msie = msie_to_kernel_1d(M_dir, L=144) 
        # Bs_mod = (1 - self.gamma_B_max) * Bs + self.gamma_B_max * K_msie
        # # Bs_mod = Bs + self.gamma_B_max * K_msie

        if not use_c: return Bs_mod, Cs

        # Cs_mod = (1 - self.gamma_C_scale*self.gamma_B_max) * Cs + self.gamma_C_scale*self.gamma_B_max  * K_msie
        Cs_mod = None
        return Bs_mod, Cs_mod

    def forward_core(self, x: torch.Tensor, h_sem: torch.Tensor, M_dir: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)         # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)                     # (B,4,d_state,L)
        Cs = Cs.float().view(B, K, -1, L)                     # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)                         # (k*d_inner) or (d_inner)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)               # (k * d)

        if self.is_mod_BC:
            Bs, Cs = self._modulate_BC(Bs, Cs, M_dir, self.use_c_mod)

        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor,  h_sem: Optional[torch.Tensor] = None, M_dir: Optional[torch.Tensor] = None, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x, h_sem, M_dir)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(self, img_size = 224, patch_size = 4, embed_dim: int = 1024, hidden_dim: int = 0, 
            drop_rate: float = 0.1, drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0, d_state: int = 16, expand: float = 2., k_sem = 10,
            a_mode='lite', shareA=True, use_entropy_for_B=False, is_mod_BC=True, use_c_mod=False,
            **kwargs, ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.ss2d = SS2D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate, a_mode=a_mode, shareA=shareA, use_entropy_for_B=use_entropy_for_B, k_sem=k_sem, is_mod_BC=is_mod_BC, use_c_mod=use_c_mod, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= 1.0 # nn.Parameter(torch.ones(hidden_dim))

        self.ln_2 = norm_layer(hidden_dim)
        self.skip_scale2 = 1.0 # nn.Parameter(torch.ones(hidden_dim))

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=embed_dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=embed_dim, norm_layer=None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        msie_out_ch = 4
        self.msie = MSIE(hidden_dim, out_ch=msie_out_ch, strip=5)
        self.conv_cat = nn.Conv2d(hidden_dim+msie_out_ch*8, hidden_dim, 1, bias=False)
        
        self.norm2 = norm_layer(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * expand)
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, drop=drop_path)

    def forward(self, input):
        # x [B, C, H, W]
        x_input, h_sem = input[0], input[1]
        # print('vss', x_input.shape, h_sem.shape)
        input_size = (x_input.shape[2], x_input.shape[3])
        x_input = self.patch_embed(x_input)
        x_input = self.pos_drop(x_input)
        B, L, C = x_input.shape

        x_input = x_input.view(B, *input_size, C).contiguous()  # [B,H,W,C]
        
        x = self.ln_1(x_input)

        x_att, M_dir = self.msie(x.permute(0, 3, 1, 2))
        x_ss2d = self.drop_path(self.ss2d(x, h_sem, M_dir))
        
        x_cat = torch.cat([x_ss2d.permute(0, 3, 1, 2), x_att], dim=1)
        x_cat = self.conv_cat(x_cat)
        x_ss2d = x_cat.permute(0, 2, 3, 1).contiguous()

        x = x_input*self.skip_scale + x_ss2d
        x = x*self.skip_scale2 + self.drop_path(self.mlp(self.norm2(x)))
      
        x = x.view(B, -1, C).contiguous()
        x = self.patch_unembed(x, input_size) # [B, C, H, W]
        return [x, h_sem]


class BridgeMap(nn.Module):
    def __init__(self, img_size=64,d_state=16,mlp_ratio=2.,n_VSSBlock=1,norm_layer=nn.LayerNorm, dropout=0.3, k_sem=10,
                 model_name='convnext_tiny.fb_in22k_ft_in1k',
                 **kwargs):
        super().__init__()
        self.single_features = True
        self.img_size = (img_size, img_size)
        self.model_name = model_name
        
        if img_size == 224:
            patchsize = 7
        elif img_size == 256:
            patchsize = 8
        else:
            patchsize = 12

        if 'tiny' in model_name:
            bk_checkpoint = 'pretrained/convnext_tiny_22k_1k_224.pth'
            self.emb_dim = 768
            D_t = 128
        elif 'base' in model_name:
            bk_checkpoint = 'pretrained/convnext_base_22k_1k_224.pth'
            self.emb_dim = 1024
            D_t = 256
        
        if '384' in model_name:
            bk_checkpoint = bk_checkpoint.replace('224', '384')
        if 'v2' in model_name:
            bk_checkpoint = bk_checkpoint.replace('_1k_', '_')
            bk_checkpoint = bk_checkpoint.replace('.pth', '.pt')
            bk_checkpoint = bk_checkpoint.replace('convnext', 'convnextv2')
        print(bk_checkpoint)

        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        # self.model = timm.create_model(model_name, pretrained=True, num_classes=0, pretrained_cfg_overlay=dict(file=bk_checkpoint))

        # build blocks
        self.is_hist = False
        self.blocks = nn.Sequential(*[VSSBlock(
            img_size=img_size, patch_size=1, embed_dim=self.emb_dim, hidden_dim=self.emb_dim, norm_layer=norm_layer, 
            attn_drop_rate=dropout, drop_path=dropout, d_state=d_state, expand=mlp_ratio, k_sem=k_sem,
            a_mode='lite', shareA=True, use_entropy_for_B=False, is_mod_BC=True, use_c_mod=False) for i in range(n_VSSBlock)])

        self.fc_text_drone = nn.Sequential(nn.Linear(512, self.emb_dim), nn.LeakyReLU(0.2))
        self.fc_text_sat = nn.Sequential(nn.Linear(512, self.emb_dim), nn.LeakyReLU(0.2))

        self.sigmoid = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d(1)

        norm_layer = get_norm_layer('layernorm2d')
        self.norm_h = norm_layer(self.emb_dim)
        self.head_n = nn.Sequential(nn.AdaptiveAvgPool2d(1), self.norm_h, nn.Flatten(1))

        D_v = 3
        self.proj_v = nn.Linear(patchsize*patchsize, D_v)
        self.norm_v = nn.LayerNorm(D_v*self.emb_dim)

        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        self.proj_t = nn.Linear(self.emb_dim, D_t)
        self.norm_t = nn.LayerNorm(D_t)

        # OSM 
        self.multihead = NormMlpClassifierHead(1024)
        self.sem_head = nn.Sequential(nn.Linear(1024, k_sem),nn.LayerNorm(k_sem))

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)
    

    def forward_vt(self, visual_feat, text_feat):
        B, C, H, W = visual_feat.shape
        visual_flat = visual_feat.view(B, C, -1)  # [B, C, H*W]
        q = text_feat  # [B, text_dim]

        att_sp = torch.matmul(q, visual_flat)
        att_sp = self.sigmoid(att_sp)
        z = visual_flat * att_sp

        visual_ch = self.avg(visual_feat).view(B, C, -1) # [B, C, 1]
        att_ch = torch.matmul(q, visual_ch) # # [B, 1, 1]
        att_ch = self.sigmoid(att_ch)
        z = att_ch * z

        # z = z + visual_flat
        z = z.view(B, C, H, W).contiguous()

        return z

    def forward_feature(self, x, x_t=None):
        # print('0', x.shape, hist.shape)
        b = x.shape[0]
        x_b = self.model.forward_features(x)

        x_b, _ = self.blocks([x_b, None])
            
        x_v = self.proj_v(x_b.view(b, self.emb_dim, -1)).view(b, -1)

        if x_t is not None:
            x_t = self.forward_vt(x_b, x_t)
            x_t = self.head_n(x_t)
            x_t = self.proj_t(x_t)

        else: return x_v
        x = torch.cat([x_v, x_t], dim=-1)
        return x.contiguous()
    

    def forward(self, img1=None, img2=None, img3=None, hist=None, query_text=[''], ref_text=['']):
        # print('input', img1.dtype, query_text.dtype)
        if img1 is not None and img2 is not None and img3 is not None:
                if len(query_text[0]) != 0 and len(ref_text[0]) != 0:
                    query_text = self.fc_text_drone(query_text.unsqueeze(1))
                    ref_text = self.fc_text_sat(ref_text.unsqueeze(1))
                else: query_text, ref_text = None, None
                image_features1 = self.forward_feature(img1, query_text)     
                image_features2 = self.forward_feature(img2, ref_text)
                image_features3 = self.forward_feature(img3, query_text)
                return image_features1, image_features2, image_features3
        elif img1 is not None and img2 is not None and img3 is None:
                query_text, ref_text = None, None
                image_features1 = self.forward_feature(img1, query_text)     
                image_features2 = self.forward_feature(img2, ref_text)
                return image_features1, image_features2    
        elif img1 is not None:
            if len(query_text[0]) != 0:
                query_text = self.fc_text_drone(query_text.unsqueeze(1))
            else: query_text = None
            image_features = self.forward_feature(img1, query_text)
            return image_features
        
        elif img2 is not None:
            if len(ref_text[0]) != 0:
                ref_text = self.fc_text_sat(ref_text.unsqueeze(1))
            else: ref_text = None
            image_features = self.forward_feature(img2, ref_text)

            return image_features
        
        elif img3 is not None:
            if len(query_text[0]) != 0:
                query_text = self.fc_text_drone(query_text.unsqueeze(1))
            else: query_text = None
            image_features = self.forward_feature(img3, query_text)
            return image_features
