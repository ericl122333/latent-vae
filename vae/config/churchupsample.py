from ml_collections import ConfigDict

def get_config():
    config = {}

    #lists sorted in descending order of resolutions. e.g, nlayers[0] is the # of stochastic layers at the highest resolution.
    config["model"] = None
    config["srmodel"] = ConfigDict({
        "resolution": 256,
        "downfactor": 8,

        "c": 64,
        "c_enc": 64,
        "c_mult": [2,2,3,3], 
        "dff_mult": [0,0,4,4],
        "zdim": 16,

        "nlayers": [3,6,8,8], 
        "enc_nlayers": [2,3,4,4], 
        "num_attention": [0,0,0,4],
        "num_attention_heads": [0,0,0,4]
    })

    config["dataset"] = ConfigDict({
        "dataset_name": 'ffhq_kl8',
        "data_dir": "{}/tfrecord",

        # Important: this is the *microbatch* size, (i.e. the batch size before accumulation, not after).
        # so reduce this number if you are running out of memory, but would like to keep a large effective batch size.
        "batch_size": 32,
        "resolution": config["srmodel"].resolution,
        "num_classes": 0,
        "shift": 127.5,
        "scale": 127.5,
        "channels": 3,
        "dtype": "uint8"
    })

    config["training"] = ConfigDict({
        "iterations": 5000000,
        "n_accums": 4,
        
        "sigma_q": 0.05,
        "rate_schedule": ("constant_per", 
            [8,4,2,1]
        ),

        "global_sr_weight": 0.25,
        "skip_threshold": 10000,

        #Save checkpoints frequently for continuing after pre-emption, but keep another subdirectory for more permanent checkpoint keeping.
        "checkpoint_dirs": ["{}/checkpoints_recent", "{}/checkpoints_permanent"],  
        "num_checkpoints": [10, 999999],
        "save_freq": [2500, 50000],

        "log_dir": "{}/logs",
        "log_freq": 500
    })

    config["optimizer"] = ConfigDict({
        "opt_type": "adam",
        "beta1": 0.9,
        "beta2": 0.9,
        "startlr": 0.0,
        "maxlr": 1e-4,
        "decay": "cosine",
        "minlr": 1e-4,
        "warmup_steps": 100,
        "decay_steps": 1200000,
        "ema_decay": 0.9998,
        "mu_dtype": 'bfloat16',
        'nu_dtype': 'bfloat16'
    })

    return ConfigDict(config)
