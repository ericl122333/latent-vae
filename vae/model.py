import torch
import torch.nn as nn
import numpy as np

from .utils import get_smoothed_variance, sample_mvn_deterministic, sample_diag_mvn, compute_mvn_kl
#from general_utils import get_resolutions, get_evenly_spaced_indices
from .nn import Conv2D, Downsample, Upsample, ResBlockFFN, ResBlockNoFFN, AttentionLayer

MIN_REMAT_RESO = 8 #inclusive

def get_resolutions(max_res, num_res):
    resos = []
    current_res = max_res
    for i in range(num_res):
        if current_res<4: current_res = 1
        resos.append(current_res)
        current_res //= 2
    return resos

def get_evenly_spaced_indices(N, K):
    #N=16, K=1: [8]
    #N=16, K=5: [3, 5, 8, 11, 13]
    #N=16, K=0: []   
    if K==0: return []

    insert_every_n = N / (K + 1)
    indices = [round(insert_every_n * i) for i in range(1, K+1)]
    return indices

class StochasticConvLayer(nn.Module):
    def __init__(self, c, c_enc, zdim, resolution, w_scale, num_classes, dff_mul=4, h_prior=False, checkpoint=False):
        super().__init__()
        self.c = c
        self.c_enc = c_enc
        self.zdim = zdim
        self.resolution = resolution
        self.w_scale = w_scale
        self.num_classes = num_classes
        self.checkpoint = checkpoint
        self.dff_mul = dff_mul
        self.h_prior = h_prior
        
        sr_lam = float(self.resolution)
        is_conv = self.resolution>=4
        ResBlock = ResBlockFFN if self.dff_mul else ResBlockNoFFN

        pchannels = c+zdim*2 if self.h_prior else c
        self.prior_block = ResBlock(c, pchannels, w_scale=1e-10, sr_lam=sr_lam, num_classes=self.num_classes, is_conv=is_conv, dff_mul=dff_mul)
        self.posterior_block = ResBlock(c, zdim*2, c_in=c, w_scale=1.0, sr_lam=sr_lam, num_classes=self.num_classes, is_conv=is_conv, dff_mul=dff_mul)
        self.actsproj = Conv2D(c+c_enc, self.c, kernel_size=1, w_scale=1.0)
        self.zproj = Conv2D(self.zdim, self.c, kernel_size=1, w_scale=w_scale, use_bias=False)
        self.shared_block = ResBlock(c, c, w_scale=w_scale, num_classes=self.num_classes, is_conv=is_conv, dff_mul=dff_mul)

    def p_sample(self, x_c, x_u=None, label=None, uncond_label=None, mweight=0.0, vweight=0.0, T=1.):
        
        zdim = self.zdim
        is_guided = (x_u is not None and self.h_prior)

        if self.h_prior:
            p_out_c, _ = self.prior_block(x_c.clone(), label)
            pmean_c, pv_unconstrained_c, h_c = p_out_c[:, :zdim, ...], p_out_c[:, zdim:zdim*2, ...], p_out_c[:, zdim*2:, ...]
            pvar_c = get_smoothed_variance(pv_unconstrained_c)
        
            if is_guided:
                p_out_u, _ = self.prior_block(x_u, uncond_label)
                pmean_u, pv_unconstrained_u, h_u = p_out_u[:, :zdim, ...], p_out_u[:, zdim:zdim*2, ...], p_out_u[:, zdim*2:, ...]
                pvar_u = get_smoothed_variance(pv_unconstrained_u)
                pmean = (pmean_c + mweight * (pmean_c - pmean_u))
                pvar = pvar_c * torch.exp(vweight * (pvar_c.log() - pvar_u.log()))
            else:
                pmean, pvar = pmean_c, pvar_c
        else:
            pmean = torch.zeros((x_c.shape[0], self.zdim, x_c.shape[2], x_c.shape[3]))
            pvar = torch.ones((x_c.shape[0], self.zdim, x_c.shape[2], x_c.shape[3]))
            h_c, _ = self.prior_block(x_c.clone(), label)

        z = sample_diag_mvn(pmean, pvar, temp=T)
        z = self.zproj(z)
        
        x_c += (z + h_c)
        x_c = self.shared_block(x_c, label)
        
        if is_guided:
            x_u += (z + h_u)
            x_u = self.shared_block(x_u, uncond_label)

        return x_c, x_u

    def forward(self, eps, x, acts, label=None):
        if label is not None: label, cf_guidance_label = label #q is always conditional, p is not necessarily
        else: cf_guidance_label = None
        zdim = self.zdim

        concatted = torch.cat((x, acts), dim=1)
        x_and_acts = self.actsproj(concatted)
        q_out, sr_loss_q = self.posterior_block(x_and_acts, label)
        qmean, qv_unconstrained = torch.chunk(q_out, 2, dim=1)   
        qvar = get_smoothed_variance(qv_unconstrained)    

        p_out, sr_loss_p = self.prior_block(x, cf_guidance_label)
        if self.h_prior:
            pmean, pv_unconstrained, h = p_out[:, :zdim, ...], p_out[:, zdim:zdim*2, ...], p_out[:, zdim*2:, ...]
            pvar = get_smoothed_variance(pv_unconstrained)
        else:
            h = p_out
            sr_loss_p *= 0.
            pmean = torch.zeros((x.shape[0], self.zdim, x.shape[2], x.shape[3]))
            pvar = torch.ones((x.shape[0], self.zdim, x.shape[2], x.shape[3]))
        kl_unweighted = compute_mvn_kl(qmean, qvar, pmean, pvar)

        z = sample_mvn_deterministic(qmean, qvar, eps=eps)  
        z = self.zproj(z)
        x += (z + h)
        x = self.shared_block(x, cf_guidance_label)
        return x, kl_unweighted, sr_loss_p+sr_loss_q

class DecoderLevel(nn.Module):
    def __init__(self, c, c_enc, zdim, nlayers, w_scale, num_classes, num_attention, head_count, current_resolution, max_resolution, c_next, h_prior, dff_mul, checkpoint=False):
        super().__init__()
        self.c = c
        self.c_enc = c_enc
        self.zdim = zdim 
        self.nlayers = nlayers
        self.w_scale = w_scale
        self.num_classes = num_classes
        self.num_attention = num_attention
        self.current_resolution = current_resolution
        self.max_resolution = max_resolution
        self.c_next = c_next

        is_conv = (self.current_resolution >= 4)
        layer_list = nn.ModuleList([])
        attention_indices = get_evenly_spaced_indices(self.nlayers, self.num_attention)

        for i in range(1, self.nlayers+1):
            layer_list.append(
                StochasticConvLayer(
                    c=self.c, 
                    c_enc=self.c_enc,
                    zdim=self.zdim, 
                    resolution=self.current_resolution,
                    w_scale=self.w_scale, 
                    num_classes=self.num_classes,
                    checkpoint=checkpoint,
                    dff_mul=dff_mul,
                    h_prior=h_prior
                )
            )
            if i in attention_indices:
                layer_list.append(AttentionLayer(self.c, checkpoint=checkpoint, num_heads=head_count))

        if self.current_resolution < self.max_resolution:
            strides = 2 if is_conv else 4
            layer_list.append(Upsample(self.c, self.c_next, strides))
        self.layer_list = layer_list

    def p_sample(self, x_c, x_u=None, label=None, uncond_label=None, mweight=1.0, vweight=1.0, T=1.):
        for layer in self.layer_list:
            if isinstance(layer, StochasticConvLayer):
                x_c, x_u = layer.p_sample(x_c, x_u, label, uncond_label, mweight=mweight, vweight=vweight, T=T) 
            else:
                x_c = layer(x_c)
                if x_u is not None:
                    x_u = layer(x_u)

        return x_c, x_u

    def forward(self, x, acts, label=None):
        shape = (x.shape[0], self.zdim , self.current_resolution, self.current_resolution)

        KLs = []
        SR_Losses = 0.
        i = 0
        for layer in self.layer_list:
            if isinstance(layer, StochasticConvLayer):
                eps = torch.randn(shape, device=x.device)
                x, kl, sr_loss = layer(eps, x, acts, label)
                KLs.append(kl)
                SR_Losses += sr_loss
                i += 1
            else:
                x = layer(x)
        return x, KLs, SR_Losses


class EncoderLevel(nn.Module):
    def __init__(self, c, nlayers, current_resolution, c_next, num_attention=0, w_scale=1.0, num_classes=0, dff_mul=4, head_count=4, checkpoint=False, label_dim=None):
        super().__init__()
        self.c = c
        self.nlayers = nlayers
        self.current_resolution = current_resolution
        self.c_next = c_next
        self.num_attention = num_attention
        self.w_scale = w_scale
        self.num_classes = num_classes
        is_conv = (self.current_resolution >= 4)

        layer_list = nn.ModuleList([])
        attention_indices = get_evenly_spaced_indices(nlayers, self.num_attention)

        ResBlock = ResBlockFFN if dff_mul else ResBlockNoFFN
        for i in range(nlayers):
            layer_list.append(
                ResBlock(
                    self.c, 
                    self.c, 
                    w_scale=self.w_scale, 
                    num_classes=self.num_classes, 
                    is_conv=is_conv,
                    checkpoint=checkpoint,
                    dff_mul=dff_mul,
                    label_dim=label_dim
                )
            )
            
            if i in attention_indices:
                layer_list.append(AttentionLayer(self.c, checkpoint=checkpoint, num_heads=head_count))
            
        if self.current_resolution > 1:
            strides = 2 if self.current_resolution > 4 else 4
            layer_list.append(Downsample(self.c, self.c_next, strides)) #note: c_next might be None
        self.layer_list = layer_list

    def forward(self, x, label=None):
        acts = None
        for layer in self.layer_list:
            if isinstance(layer, Downsample):
                acts = x.clone()
            x = layer(x, label)

        if acts is None: acts = x
        return x, acts
        

class Encoder(nn.Module):
    def __init__(self, c, c_mult, nlayers, resolution, num_attention, num_classes, checkpoint_min_resolution, dff_mult, num_attention_heads, label_dim, data_dim):
        super().__init__()
        self.c = c
        self.c_mult = c_mult
        self.nlayers = nlayers
        self.resolution = resolution
        self.num_attention = num_attention
        self.num_classes = num_classes
        self.label_dim = label_dim
        self.data_dim = data_dim
    
        C = [self.c * mult for mult in self.c_mult]
        nlayers = self.nlayers
        w_scale = 1/np.sqrt(sum(nlayers))
        num_resolutions = len(nlayers)
        resolutions = get_resolutions(self.resolution, num_resolutions)

        self.in_proj = Conv2D(self.data_dim, C[0], kernel_size=3)
        self.encoder_levels = nn.ModuleList([])
        for i in range(num_resolutions):
            c_next = None
            if i<num_resolutions-1:
                if C[i] != C[i+1]:
                    c_next = C[i+1] 
            
            self.encoder_levels.append(
                EncoderLevel(
                    C[i],
                    nlayers[i],
                    resolutions[i],
                    c_next,
                    self.num_attention[i],
                    w_scale,
                    self.num_classes,
                    checkpoint=(resolution >= checkpoint_min_resolution),
                    dff_mul=dff_mult[i],
                    head_count=num_attention_heads[i],
                    label_dim=label_dim
                )
            )
    
    def forward(self, img, label=None):
        x = self.in_proj(img)
        acts = []
        for layer in self.encoder_levels:
            x, acts_i = layer(x, label)
            acts.append(acts_i)
        
        return acts

class VAE(nn.Module):
    def __init__(self, c, c_enc, c_mult, zdim, nlayers, enc_nlayers, datadim, num_attention, num_attention_heads, resolution, num_classes, dff_mult, h_prior, checkpoint_min_resolution=32):
        super().__init__()
        self.c = c
        self.c_enc = c_enc
        self.c_mult = c_mult
        self.zdim = zdim
        self.nlayers = nlayers
        self.enc_nlayers = enc_nlayers
        self.num_attention = num_attention
        self.num_attention_heads = num_attention_heads
        self.resolution = resolution
        self.num_classes = num_classes
        self.dff_mult = dff_mult
        self.h_prior = h_prior
        self.datadim = datadim

        self.cond = self.num_classes > 0
        resolutions = get_resolutions(self.resolution, len(self.nlayers))
        C = [self.c * mult for mult in self.c_mult]
        C_enc = [self.c_enc * mult for mult in self.c_mult]
        w_scale = 1/np.sqrt(sum(self.nlayers))

        self.encoder = Encoder(self.c_enc, self.c_mult, self.enc_nlayers, self.resolution, 
            self.num_attention, self.num_classes, checkpoint_min_resolution=checkpoint_min_resolution, 
            dff_mult=dff_mult, num_attention_heads=num_attention_heads, label_dim=C[-1], data_dim=datadim)
        
        if self.cond:
            self.embed = nn.Embedding(self.num_classes+1, C[-1])

        self.initial_x = nn.Parameter(torch.zeros([1, C[-1], resolutions[-1], resolutions[-1]])) #this is a permuted parameter. be sure to catch this one and permute it.

        layer_list = []
        for i in reversed(range(len(resolutions))):
            c_next = C[i-1] if i>0 else None
            if c_next == C[i]: c_next = None

            layer_list.append(
                DecoderLevel(
                    c=C[i],
                    c_enc=C_enc[i],
                    zdim=self.zdim,
                    nlayers=self.nlayers[i],
                    w_scale=w_scale,
                    num_classes=self.num_classes,
                    num_attention=self.num_attention[i],
                    current_resolution=resolutions[i],
                    max_resolution=resolutions[0],
                    c_next=c_next,
                    checkpoint=(resolutions[i] >= checkpoint_min_resolution),
                    head_count=num_attention_heads[i],
                    h_prior=h_prior,
                    dff_mul=dff_mult[i]

                )
            )
        self.layer_list = nn.ModuleList(layer_list)

        self.outproj = Conv2D(C[0], self.datadim * 2, 1)

    @torch.no_grad()
    def p_sample(self, num=10, label=None, mweight=1.0, vweight=1.0):

        if self.num_classes:
            uncond_label = torch.ones_like(label) * self.num_classes
            label = self.embed(label)
            uncond_label = self.embed(uncond_label)
        else:
            uncond_label = None

        is_guided = (mweight != 0 or vweight != 0)
        x_c = torch.tile(self.initial_x, [num, 1, 1, 1])
        x_u = torch.tile(self.initial_x, [num, 1, 1, 1]) if is_guided else None
        
        if isinstance(mweight, float): mweight = [mweight] * len(self.nlayers)
        if isinstance(vweight, float): vweight = [vweight] * len(self.nlayers)

        #update cond'l and uncond'l hidden states at each stochastic layer.
        for i, decoderlevel in enumerate(self.layer_list):
            x_c, x_u = decoderlevel.p_sample(x_c, x_u, label, uncond_label, mweight=mweight[i], vweight=vweight[i])
            if torch.isnan(x_c.var()):
                raise RuntimeError() #TODO remove
        x_c = self.outproj(x_c)
        return x_c[:, :self.datadim, ...] #ignore variance output of the model as it's not needed for visualizing images.
    

    def forward(self, img, label=None):
        #randomly drop class label in the prior w/ 10% probability to enable CFG
        if self.cond:
            uncond_label = torch.full_like(label, self.num_classes)
            mask = torch.greater(torch.rand(label.shape), 0.9).int()
            cf_guidance_label = label*(1-mask) + mask*uncond_label

            label = self.embed(label)
            cf_guidance_label = self.embed(cf_guidance_label)
            generator_label = (label, cf_guidance_label)
        else: 
            label, generator_label = None, None

        x = torch.tile(self.initial_x, [img.shape[0], 1, 1, 1])

        acts = self.encoder(img, label=label)
        KLs = []
        SR_Losses = 0.

        for i, decoderlevel in enumerate(self.layer_list):
            j = -i+len(self.nlayers)-1
            x, kls, sr_losses = decoderlevel(x, acts[j], generator_label)
            KLs.extend(kls)
            SR_Losses += sr_losses
          
        x = self.outproj(x)
        return x, KLs, SR_Losses