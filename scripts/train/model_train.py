# https://docs.ultralytics.com/modes/train/

import os
from ultralytics import RTDETR
import argparse
from pathlib import Path
import wandb
from root import ROOT
from datetime import datetime

root_path = ROOT
wandb.login()

def on_fit_epoch_end(trainer):
    log_data = {
        "epoch": trainer.epoch + 1,
        **{k: v for k, v in trainer.metrics.items() if k != "step"}
    }
    wandb.log(log_data, step=trainer.epoch + 1)

def main(file_name, epoch, batches, imgsz) :
    wandb.init(       
        notes = f"Training RT-DETR on {file_name} dataset",
        name = f"RT_DETR_V2_X-{datetime.now().strftime('%m%d_%H%M')}",
        project = "Minimap Detection",
        config = {
            "model": "rtdetr-x.yaml",
            "epochs": epoch,
            "batch_size": batches,
            "imgsz": imgsz,
            "optimizer": "AdamW",
            "lr0": 0.0002,
            "lrf": 0.05,
            "cos_lr": True,
            "weight_decay": 0.0003,
            "warmup_epochs": 10,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.003,
            "augmentation": "hsv = [0.01, 0.2, 0.1]",
            "nbs": 128
        }
    )
    wandb.define_metric("metrics/*", step_metric="epoch")

    model = RTDETR(os.path.join(root_path, 'scripts', 'train', 'rtdetr-x.yaml'))
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    datapath = root_path / 'data' / 'synthetics' / file_name /'config.yaml'
    project_path = root_path / 'results' / 'model' / 'pretrain'

    results = model.train(
        data = datapath,
        epochs = epoch,                 
        batch = batches,                   
        imgsz = imgsz,

        # learning setting
        device = '0',
        workers = 8,
        optimizer = 'AdamW',
        patience = 10,

        # learning rate
        lr0 = 0.0002,
        lrf = 0.05,
        cos_lr = True,
        weight_decay = 0.0003,

        # warmup
        warmup_epochs = 10,
        warmup_momentum = 0.8,
        warmup_bias_lr = 0.003,

        # augmentation deactivate
        hsv_h = 0.01,
        hsv_s = 0.2,
        hsv_v = 0.1,
        degrees = 0,
        translate = 0,
        scale = 0,
        shear = 0,
        perspective = 0,
        flipud = 0,
        fliplr = 0,
        mosaic = 0,
        erasing = 0,
        auto_augment = None,

        # Pretrained
        pretrained = False,

        # cls_loss explosion proof
        nbs = 128,

        # project name
        project= project_path,
        name=""
    )
    wandb.finish()
    return results

if __name__ == '__main__' :
    """
    python -m model_train --file_name lol_minimap_256_sample --epoch 100 --batches 32 --imgsz 256
    """
    parser = argparse.ArgumentParser(description="Train the model with the given parameters.")

    parser.add_argument('--file_name', type=str, required=True, help='Name of the file')
    parser.add_argument('--epoch', type=int, default=4, help='Number of epochs (default: 4)')
    parser.add_argument('--batches', type=int, default=4, help='Number of batches (default: 4)')
    parser.add_argument('--imgsz', type=int, default=256, help='Image size (default: 256)')

    args = parser.parse_args()

    main(args.file_name, args.epoch, args.batches, args.imgsz)
