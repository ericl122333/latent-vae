import torch
import numpy as np

from dae.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from dae.dae_util import instantiate_from_config
from omegaconf import OmegaConf
from vae.model import VAE

class Pipeline(torch.nn.Module):
    def __init__(self, vae_config, dae_config):
        super().__init__()
        #vae_config is an ml-collections dictionary
        #dae_config is an OmegaConf dictionary

        self.dae_config = dae_config
        self.dae = instantiate_from_config(self.dae_config)

        self.vae_config = vae_config
        self.vae = VAE(**self.vae_config.model)
        

    def merge_individual_ckpts_into_single(self, save_path, vae_ckpt_path, dae_ckpt_path):
        self.dae.init_from_ckpt(dae_ckpt_path)
        self.vae.load_state_dict(torch.load(vae_ckpt_path))
        torch.save(self.state_dict(), save_path)

    def load_from_ckpt(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path))
    
    def sample(self, num=10, label=None, mweight=1.0, vweight=1.0, dae_batchsize=2):
        output_ims = []

        with torch.no_grad():
            latents = self.vae.p_sample(num, label, mweight=mweight, vweight=vweight)
            latents = (latents * self.vae_config.dataset.scale) + self.vae_config.dataset.shift

            for i in range(num // dae_batchsize):
                lats = latents[i*dae_batchsize : i*dae_batchsize+dae_batchsize]
                ims = self.dae.decode(lats)
                ims = ims.float().cpu().permute(0, 2, 3, 1)
                output_ims.append(ims)
        
        output_ims = np.concatenate(tuple(output_ims), axis=0)
        output_ims = output_ims * 0.5 + 0.5
        output_ims = np.clip(output_ims, 0., 1.) * 255
        output_ims = output_ims.astype('uint8')
        return output_ims
