from ml_collections import ConfigDict

def get_config():
    config = {}

    config["model"] = ConfigDict({
        "resolution": 32,

        "c": 192,
        "c_enc": 192,
        "c_mult": [1,1,1,1,1], 
        "dff_mult": [4,4,4,4,4],
        "datadim": 4,
        "zdim": 16,

        #sorted in descending order of resolutions. e.g, nlayers[0] is the # of stochastic layers at the highest (32x32) resolution.
        "nlayers": [8,8,7,6,6], 
        "enc_nlayers": [4,4,4,3,3], 
        "num_attention": [4,4,4,0,0],
        "num_attention_heads": [4,8,8,0,0],
        "num_classes": 0,
        "h_prior": False
    })

    config["dataset"] = ConfigDict({
        "dataset_name": 'bedroom_kl8',
        "data_dir": "{}/tfrecord",

        # Important: this is the *microbatch* size, (i.e. the batch size before accumulation, not after).
        # so reduce this number if you are running out of memory, but would like to keep a large effective batch size.
        "batch_size": 128,
        "resolution": config["model"].resolution,
        "num_classes": config["model"].num_classes,
        "shift": 0.25,
        "scale": 4.1,
        "channels": 4,
        "dtype": "float32"
    })

    config["training"] = ConfigDict({
        "iterations": 10000000,
        "n_accums": 1,
        
        "sigma_q": 0.07,
        "rate_schedule": ("shifted_exp", 
            {"scale": 1000, "shift": 10}
        ),

        "global_sr_weight": 0.1,
        "skip_threshold": 300,

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
        "maxlr": 2e-4,
        "decay": "cosine",
        "minlr": 2e-4,
        "warmup_steps": 100,
        "decay_steps": 1200000,
        "ema_decay": 0.9999,
        "mu_dtype": 'bfloat16',
        'nu_dtype': 'bfloat16'
    })

    return ConfigDict(config)
