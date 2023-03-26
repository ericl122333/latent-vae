import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from .utils import transposed_matmul, maybe_remat

def Conv2D(c_in, c_out, kernel_size, w_scale=1., strides=1, use_bias=True):
    padding = kernel_size//2 #dimension preserving padding for all odd kernel sizes.
    conv = nn.Conv2d(
        c_in, c_out,
        kernel_size=(kernel_size, kernel_size),
        stride=(strides, strides),
        bias=use_bias,
        padding=padding
    )
    torch.nn.init.xavier_uniform_(conv.weight)
    conv.weight.data.mul_(w_scale)
    return conv

class ConvSR(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, w_scale=1.0, sr_lam=1.0):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.w_scale = w_scale
        self.sr_lam = sr_lam

        self.conv = Conv2D(self.c_in, self.c_out, self.kernel_size, w_scale=self.w_scale)
        self.singular_vectors = torch.randn([1, self.c_out])
    
    def forward(self, x):
        out = self.conv(x)

        w = self.conv.weight.data
        w = torch.reshape(w, [-1, self.c_out]) #[C, C, k, k] -> [k** 2 * C, C] 

        self.singular_vectors = self.singular_vectors.to(w.device)
        v_unnormalized = transposed_matmul(self.singular_vectors, w, perm=[1, 0])
        v = v_unnormalized / torch.linalg.norm(v_unnormalized, ord=2)
        u_unnormalized = torch.matmul(v, w)

        self.singular_vectors = u_unnormalized / torch.linalg.norm(u_unnormalized, ord=2)

        sigma = transposed_matmul(
            torch.matmul(v, w), self.singular_vectors, perm=[1, 0]
        )
        return out, torch.squeeze(sigma * self.sr_lam)

class Downsample(nn.Module):
    def __init__(self, c_in, c_out, strides):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.has_conv = (c_in is not None and c_out is not None)
        self.strides = strides
        if self.has_conv:
            self.conv_layer = Conv2D(c_in, c_out, 1)

    def forward(self, x, label=None):
        B, C, H, W = x.shape
        strides = self.strides
        x = F.avg_pool2d(x, (strides, strides), (strides, strides))

        if self.has_conv:
            x = self.conv_layer(x)
        return x

class Upsample(nn.Module):
    def __init__(self, c_in, c_out, strides):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.has_conv = (c_in is not None and c_out is not None)
        self.strides = strides
        if self.has_conv:
            self.conv_layer = Conv2D(c_in, c_out, 1)

    def forward(self, x, label=None):
        B, C, H, W = x.shape
        x = F.interpolate(x, size=[H * self.strides, W * self.strides], mode="nearest")
        if self.has_conv:
            x = self.conv_layer(x)
        return x

class ResBlockNoFFN(nn.Module):
    def __init__(self, c, c_out, c_in=None, w_scale=1.0, sr_lam=None, num_classes=0, is_conv=True, checkpoint=False, dff_mul=None, label_dim=None):
        super().__init__()
        self.c = c
        self.c_out = c_out
        self.c_in = c_in or c

        self.w_scale = w_scale
        self.sr_lam = sr_lam
        self.num_classes = num_classes
        self.is_conv = is_conv
        self.checkpoint = checkpoint

        if self.is_conv: ksize = 3
        else: ksize = 1

        if label_dim is None:
            label_dim = c
        if self.num_classes > 0:
            self.classproj = nn.Linear(label_dim, self.c)

        self.conv1 = Conv2D(self.c_in, self.c, ksize)

        if self.sr_lam is not None:
            self.conv2 = ConvSR(self.c, self.c_out, ksize, w_scale=self.w_scale)
        else:
            self.conv2 = Conv2D(self.c, self.c_out, ksize, w_scale=self.w_scale)
    
    @maybe_remat
    def forward(self, inputs, label=None):
        x = inputs.clone()

        if self.num_classes > 0: 
            label = self.classproj(label)
            print(label.shape, label[:, :, None, None].shape, label.unsqueeze(-1).unsqueeze(-1).shape)
            x += label[:, :, None, None] #shape was changed from [:, none, none, :]

        x = self.conv1(F.silu(x))
        if self.sr_lam is not None:
            x, sr_loss = self.conv2(F.silu(x))            
            return x, sr_loss
        else:
            x = self.conv2(F.silu(x))            
            if self.c==self.c_out:  
                x += inputs
            return x
    
class ResBlockFFN(nn.Module):
    def __init__(self, c, c_out, c_in=None, w_scale=1.0, sr_lam=None, num_classes=0, is_conv=True, dff_mul=4, label_dim=None, checkpoint=False):
        super().__init__()
        self.c = c
        self.c_out = c_out
        self.c_in = c_in or c

        self.w_scale = w_scale
        self.sr_lam = sr_lam
        self.num_classes = num_classes
        self.is_conv = is_conv
        self.checkpoint = checkpoint
        self.dff_mul = dff_mul

        if self.is_conv: ksize = 3
        else: ksize = 1

        if label_dim is None:
            label_dim = c
        if self.num_classes > 0:
            self.classproj = nn.Linear(label_dim, self.c)
        self.conv1 = Conv2D(self.c_in, self.c, ksize)
        self.conv2 = Conv2D(self.c, self.c, ksize)
        self.dense1 = Conv2D(self.c, self.c * self.dff_mul, 1)

        if self.sr_lam is not None:
            self.dense2 = ConvSR(self.c * self.dff_mul, self.c_out, 1, w_scale=self.w_scale)
        else:
            self.dense2 = Conv2D(self.c * self.dff_mul, self.c_out, 1, w_scale=self.w_scale)
    
    @maybe_remat
    def forward(self, inputs, label=None):
        x = inputs.clone()

        if self.num_classes > 0: 
            label = self.classproj(label)
            x += label[:, :, None, None] #shape was changed from [:, none, none, :]

        x = self.conv1(F.silu(x))
        x = self.conv2(F.silu(x))

        x = self.dense1(F.silu(x))
        if self.sr_lam is not None:
            x, sr_loss = self.dense2(F.silu(x))            
            return x, sr_loss
        else:
            x = self.dense2(F.silu(x))            
            if self.c==self.c_out:  
                x += inputs
            return x

class AttentionLayer(nn.Module):
    def __init__(self, c, num_heads=1, checkpoint=False):
        super().__init__()
        self.c = c
        self.num_heads = num_heads
        self.dhead = c // num_heads
        self.checkpoint = checkpoint

        # with 1 head, we only use a q-projection, because we fuse Q and K projections together, and the V and output projections are fused together too
        # note that this doesn't limit the expressivity, b/c when there's one head:
        # (x @ Wq) @ (x @ Wk).T = (x @ Wq) @ (Wk.T @ x.T) = x @ (Wq @ Wk.T) @ x.T -> notice how only one weight matrix is needed
        if num_heads==1:
            self.qproj = nn.Linear(c, c)
        else:
            self.qkvproj = nn.Linear(c, c*3)
        
        self.outproj = nn.Linear(c, c)
        self.outproj.weight.data.fill_(0.0)
    
    @maybe_remat
    def forward(self, x, label=None):
        B, C, H, W = x.shape

        x = torch.reshape(x, [B, C, H*W]).permute(0, 2, 1)
        _rescon = x.clone()

        if self.num_heads == 1:
            q = self.qproj(x)
            qk = transposed_matmul(q, x, perm=[0, 2, 1]) / np.sqrt(self.c)
            attention_weights = F.softmax(qk, dim=-1)
            attn_out = torch.matmul(attention_weights, x)
        else:
            qkv = self.qkvproj(x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)

            q = torch.reshape(q, [B, H*W, self.num_heads, self.dhead]) 
            k = torch.reshape(k,  [B, H*W, self.num_heads, self.dhead]) 
            v = torch.reshape(v,  [B, H*W, self.num_heads, self.dhead]) 
            qk = torch.einsum('bqhd,bkhd->bhqk', q, k) / np.sqrt(self.dhead)

            attention_weights = F.softmax(qk, dim=-1)
            attn_out = torch.einsum('bhqk,bkhd->bqhd', attention_weights, v)
            attn_out = torch.reshape(q, [B, H*W, self.num_heads * self.dhead])

        x = self.outproj(attn_out)
        x = x + _rescon
        x = torch.reshape(x.permute(0, 2, 1), [B, C, H, W])
        return x