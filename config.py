from pathlib import Path

def get_config():
    return {
        # ConvNext model params (ConvNext T here)
        "in_channels": 3,
        "num_classes": 101,
        "block_sizes": [96, 192, 384, 768],
        "depths": [3, 3, 9, 3],
        "drop_path_rate": 0.1,
        # Training params
        "batch_size": 256,
        "num_epochs": 100,
        "lr": 4e-3 * (256/4096),
        "preload": None,
        "dataset_folder": "dataset",
        "model_folder": "weights",
        "model_basename": "convnextmodel_",
        "experiment_name": "runs/convNextmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}"
    return str(Path(".")/model_folder/model_filename)