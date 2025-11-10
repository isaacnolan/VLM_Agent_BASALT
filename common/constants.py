"""Shared constants."""

# MineRL action space configuration
ACTION_TRANSFORMER_KWARGS = {
    "camera_binsize": 2,
    "camera_maxval": 10,
    "camera_mu": 10,
    "camera_quantization_scheme": "mu_law",
}

# Action keys for MineRL
ACTION_KEYS = [
    'attack', 'back', 'forward', 'jump', 'left', 'right',
    'sneak', 'sprint', 'use', 'drop', 'inventory',
    'hotbar.1', 'hotbar.2', 'hotbar.3', 'hotbar.4', 'hotbar.5',
    'hotbar.6', 'hotbar.7', 'hotbar.8', 'hotbar.9',
    'camera', 'ESC'
]
