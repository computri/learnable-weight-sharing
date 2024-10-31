from yacs.config import CfgNode as CN

def get_cfg_defaults():
    cfg = CN()

    cfg.batch_size = -1
    cfg.epochs = -1
    cfg.seed = -1
    cfg.learning_rate = 0.0
    cfg.learning_rate_reps = 0.0
    cfg.norm_scaler = 0.0
    cfg.entropy_scaler = 0.0
    cfg.cuda = -1
    cfg.group = ""
    cfg.transform_labels = ""
    cfg.log_wandb = -1
    cfg.n_group = -1
    cfg.n_iter = -1
    cfg.hidden_channels = -1
    cfg.model = ""
    cfg.init_mode = ""
    cfg.visualize = -1
    cfg.augment = -1
    cfg.num_layers = -1
    cfg.kernel_size = -1
    cfg.fix_identity = -1
    cfg.dataset = ""

    return cfg