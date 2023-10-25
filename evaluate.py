import torchvision
import torch
from pipeline import Pipeline
from omegaconf import OmegaConf
from ml_collections import ConfigDict
import argparse
import PIL
from tensorflow.io import gfile
import time
import os
import warnings
import numpy as np

from vae.config.bedroom import get_config as bedroom_config
from vae.config.church import get_config as church_config
from vae.config.imagenet256_kl8 import get_config as imagenet_config

INFO = {
    "bedroom": ConfigDict({
        "dae_config": "dae/config/bedroom.yaml",
        "vae_config": bedroom_config(),
        "weights_name": "./models/bedroom_pipeline.pt",
        "weights_id": "1--uqW9S8-tbMJG8whHDx1qPJeH5W0j9r",
    }),
    "church": ConfigDict({
        "dae_config": "dae/config/church.yaml",
        "vae_config": church_config(),
        "weights_name": "./models/church_pipeline.pt",
        "weights_id": "1-174o5MywJYhhOcYI6yzMfUP5DvnlPcK",
    }),
    "imagenet": ConfigDict({
        "dae_config": "dae/config/imagenet.yaml",
        "vae_config": imagenet_config(),
        "weights_name": "./models/inet_pipeline.pt",
        "weights_id": "1-Td-danBSRX4IhlXAD_CtgFARbu4C9hH",
    }),
}

parser = argparse.ArgumentParser(description='Run sampling.')
parser.add_argument('model_name', type=str, help='name of the model')
parser.add_argument('--n_sample', type=int, default=12, help='number of samples')
parser.add_argument('--n_row', type=int, default=6, help='number of images displayed in each row')
parser.add_argument('--auto_download', default="True", choices=['True','true', 'False', 'false'], help='whether to auto download')

parser.add_argument('--label', type=str, default=None, help='class label for the generated samples. default is random classes')
parser.add_argument('--mweight', type=float, default=1.5, help='mean guidance weight for imagenet model. use 0.0 for unguided, NOT 1.0.')
parser.add_argument('--vweight', type=float, default=3.0, help='variance guidance weight for imagenet model. use 0.0 for unguided, NOT 1.0.')
parser.add_argument('--dae_batchsize', type=int, default=2, help='batch size for DAE decoding')
parser.add_argument('--save_format', type=str, default='grid', choices=['grid', 'npz'], 
                    help="Either 'grid' or 'npz'. Determines whether to save results as a grid of images (default, best for <= 100 images), or as an .npz file (for evaluation).")

args = parser.parse_args()

def save_images(imgs_grid, filename):
    if not (filename.startswith("gs://") or filename.startswith("gcs://")):
        image = PIL.Image.fromarray(imgs_grid)
        image.save(filename)
    else:
        #save a temporary version locally, then move to GCS remote storage, then delete the local copy.
        image = PIL.Image.fromarray(imgs_grid)
        image.save(f"./results/tmp_figure.png")
        gfile.copy(f"./results/tmp_figure.png", filename)
        time.sleep(1.5)
        gfile.remove(f"./results/tmp_figure.png")
                
def main():
    args = parser.parse_args()
    model_name = args.model_name.lower()
    if model_name in INFO:
        config = INFO[model_name]
    else:
        raise ValueError(f"""model_name argument should be one of "imagenet", "church", or "bedroom", but got {args.model_name}""")
    
    vae_config = config.vae_config
    dae_config = OmegaConf.load(config.dae_config)["model"]
    pipe = Pipeline(vae_config, dae_config)
    if torch.cuda.is_available():
        pipe = pipe.to('cuda')

    print(torch.stack([w.data.square().sum() for w in pipe.parameters()]).sum())
    try:
        pipe.load_from_ckpt(config.weights_name)
    except FileNotFoundError:
        print(f"weights not found in location {config.weights_name}")
        e = RuntimeError("You will need to manually download the weights as specified in the readme.")
        try:
            if args.auto_download.lower() != "false":
                print("trying to auto-download the weights...")
                import gdown
                if not os.path.isdir("./models"):
                    os.mkdir("./models")
                gdown.download(id=config.weights_id, output=config.weights_name, quiet=False)
                pipe.load_from_ckpt(config.weights_name)
            else:
                raise e
        except:
            raise e
        

    print(torch.stack([w.data.square().sum() for w in pipe.parameters()]).sum())
    if pipe.vae.num_classes:
        device = next(pipe.parameters()).device
        if args.label is None:
            label = torch.randint(size=[args.n_sample], low=0, high=pipe.vae.num_classes).to(device)
        else:
            label = (torch.ones([args.n_sample]) * int(args.label)).int().to(device)
    else:
        label = None
    
    samples = pipe.sample(num=args.n_sample, label=label, mweight=float(args.mweight), vweight=float(args.vweight), dae_batchsize=args.dae_batchsize)
    
    if not os.path.isdir("./results"):
        os.mkdir("./results")
    
    ext = "png" if args.save_format.lower() == "grid" else "npz"
    label_string = "uncond" if args.label is None else args.label

    samples_identifier = f"{len(gfile.glob(f'./results/*.{ext}'))}_{label_string}"
    if model_name == 'imagenet':
        samples_identifier += f"_m{args.mweight}_v{args.vweight}"
    samples_path = os.path.join("./results/", f"samples_{samples_identifier}.{ext}")
    
    print("created samples of shape: ", samples.shape)
    num = samples.shape[0]
    if args.save_format.lower() == "grid":
        if len(samples) > 200:
            warnings.warn("you have more than 200 samples in your grid, which is too large. truncating to the first 200 samples.")
            samples = samples[:200]
            num = 200
        samples = torchvision.utils.make_grid(torch.tensor(samples).permute(0,3,1,2), nrow=args.n_row)
        samples = samples.permute(1,2,0).numpy()
        save_images(samples, samples_path)
    else:
        np.savez("tmpfile.npz", arr0=samples)
        gfile.copy("tmpfile.npz", samples_path)
        time.sleep(3.0)
        gfile.remove("tmpfile.npz")

    print(f"Saved {num} samples to {samples_path}")

if __name__ == '__main__':
    main()
    
