# Encoder + expander architecture
MODEL_CONFIG = {
    "encoder_channels": (64, 128, 256),
    "expander_hidden":  512,
    "expander_out":     512,
}

# Training
TRAIN_CONFIG = {
    "batch_size": 256,
    "epochs":     20,
    "lr":         3e-4,
}

# VICReg loss coefficients (paper defaults)
LOSS_CONFIG = {
    "lam":   25.0,   # invariance
    "mu":    25.0,   # variance
    "nu":    1.0,    # covariance
    "gamma": 1.0,    # target std
    "eps":   1e-4,
}

# Ablation runs: (name, use_var, use_cov) — demonstrates the effect of each term
ABLATIONS = [
    ("vicreg_full",  True,  True),
    ("no_variance",  False, True),
    ("no_covariance", True, False),
]
