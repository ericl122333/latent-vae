import os
import numpy as np
import torch
import time
from tensorflow.io import gfile
from torch.utils.checkpoint import CheckpointFunction
import inspect

#be careful! if you set "keep" to a small number, it will remove checkpoints in that directory without warning.
def save_checkpoint(state, ckpt_dir=None, ckpt_path=None, accelerator=None, keep=99):
    assert isinstance(ckpt_dir, str) != isinstance(ckpt_path, str), "Provide either ckpt_dir OR ckpt_path (don't provide both)."
    
    if ckpt_path is None:
        ckpt_paths = sorted(gfile.glob(f'{ckpt_dir}/*.pt'))
        while len(ckpt_paths) > keep:
            gfile.remove(ckpt_paths[0])
            time.sleep(1.0)
            ckpt_paths = sorted(gfile.glob(f'{ckpt_dir}/*.pt'))

        step = state.get_iteration() if "get_iteration" in dir(state) else "latest"
        ckpt_path = os.path.join(ckpt_dir, f'ckpt_{step}.pt')

    with gfile.GFile(ckpt_path, "wb") as f:
        if accelerator is not None:
            accelerator.save(state.state_dict(), f)
        else:
            torch.save(state.state_dict(), f)

#works for either model or TrainState
def restore_checkpoint(empty_state, ckpt_dir=None, step=None, ckpt_path=None):
    if ckpt_path is None:
        assert ckpt_dir is not None, "must provide either ckpt_path or ckpt_dir"
        if step is None:
            ckpt_paths = sorted(gfile.glob(f'{ckpt_dir}/*.pt'))
            if len(ckpt_paths) == 0:
                print(f"No existing checkpoints found in {ckpt_dir}")
                return empty_state
            else:
                ckpt_path = ckpt_paths[-1]
        else:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_{step}.pt")
    

    state_dict = torch.load(ckpt_path)
    restored_state = empty_state.load_state_dict(state_dict)
    return restored_state


def calc_weight_for_kl(x, a, b, c):

    mask1 = torch.less(x, a).float()
    mask2 = torch.logical_and(torch.greater_equal(x, a), torch.less(x, b)).float()
    mask3 = torch.logical_and(torch.greater_equal(x, b), torch.less(x, c)).float()
    mask4 = torch.greater_equal(x, c).float()

    piece1 = x/a
    piece2 = 1.
    piece3 = (x-b) / (c-b) + 1.
    piece4 = 2.
    weighting_function_output = piece1*mask1 + piece2*mask2 + piece3*mask3 + piece4*mask4
    return torch.maximum(weighting_function_output, 0.1 * torch.ones_like(weighting_function_output)) #the weight should never be zero.

def weighted_kl(unweighted_kl, sum_of_kl, lower_percentage, upper_percentage):
    a = lower_percentage * sum_of_kl * 0.01  #the 0.01 is because its a percentage
    b = upper_percentage * sum_of_kl * 0.01 
    w = calc_weight_for_kl(unweighted_kl, a, b, a+b).detach()#.requires_grad_(True) # is requires grad needed?
    adjusted_kl = unweighted_kl * w
    return adjusted_kl.sum()

#VAE related utils
def get_smoothed_variance(var_unconstrained):
    return (1/0.693) * torch.log(1 + torch.exp(0.693 * var_unconstrained))

def transposed_matmul(a, b, perm):
    b = torch.permute(b, dims=perm)
    return torch.matmul(a, b)

def sample_diag_mvn(mean, var, temp=1., eps=None):
    if eps is None:
        eps = torch.randn_like(mean)
    return mean + eps * var.sqrt() * temp

sample_mvn_deterministic = sample_diag_mvn

def compute_mvn_kl(mu1, var1, mu2, var2, rdim=(-1, -2, -3)):
    #workaround for float value of sigma_q, maybe fix later.
    if isinstance(var1, float):
        logvar1 = np.log(var1)
    else:
        logvar1 = var1.log()

    C = 0.5 * (var2.log() - logvar1 - 1)
    kl = 0.5 * (var1 + (mu1 - mu2) **2) / var2 + C
    kl = kl.sum(dim=rdim)
    return kl

def count_params(model):
    return sum([np.prod(p.shape) for p in model.parameters()])

# torch related utils
def compute_global_norm(grads):
    norms = torch.stack([
        torch.linalg.norm(g) for g in grads
    ], dim=-1)
    return torch.linalg.norm(norms)


"""
a decorator for applying gradient checkpointing to a module's forward method, similar to flax's nn.remat decorator.
it will perform gradient checkpointing for the module unless the module has an attribute "checkpoint" that evaluates to True
it does not, however, allow for wrapping a class within the remat call, e.g. remat(nn.Linear)(c_in, c_out)
example usage:
class MyModule(nn.Module):
    def __init__(*args, **kwargs):
        #init

    @maybe_remat
    def forward(*args, *kwargs)
        #do forward
"""
def maybe_remat(forward_method):
    #TODO: fix this error: https://discuss.pytorch.org/t/getting-this-warning-output-0-of-backwardhookfunctionbackward-is-a-view-and-is-being-modified-inplace/122766
    return forward_method
    
    def checkpointed_method(*args, **kwargs):
        module = args[0]
        module_parameters = module.parameters()
        if not hasattr(module, "checkpoint") or not module.checkpoint:
            #do the regular forward method if module.checkpoint is False, or if it doesn't have property checkpoint
            return forward_method(*args, **kwargs)

        tuple(module_parameters) #workaround to prevent invalid number of arguments error from being thrown. 
        input_args = list(args)
        #CheckpointFunction expects a tuple of input_args, so if someone calls a forward() method with keyword arguments,
        #these values will not be passed in as they are in kwargs, not args. So, we add 
        all_args = inspect.signature(forward_method).parameters
        argnames, argvals = list(all_args.keys()), list(all_args.values())
        for argnum in range(len(input_args), len(all_args)):
            argname = argnames[argnum]
            if argname in kwargs.keys():
                input_args.append(kwargs.pop(argname))
            else:
                default_arg = argvals[argnum].default
                if default_arg == inspect._empty:
                    raise RuntimeError(f"forward method missing positional argument {argname}")
                input_args.append(default_arg)

        if len(kwargs) > 0:
            raise RuntimeError(f"forward method received unexpected keyword arguments: {list(kwargs.keys())}")

        args_and_parameters = tuple(input_args) + tuple(module_parameters)
        return CheckpointFunction.apply(forward_method, len(input_args), *args_and_parameters)

    return checkpointed_method