import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter

from config import get_config, get_weights_file_path
from dataset import load_dataloader
from model import get_model

from pathlib import Path

from tqdm import tqdm

# Again, paper mentioned some specific training steps like using Adam Optim these are just changes to adapt modern practices
# The paper said 300 epochs but since this demonstration used much smaller dataset, sticking with 75-100 epochs
# The paper also included Exponential Moving Average which is helpful for large datasets and deeper models but ConvNeXt tiny it might not be, so skipped
# Paper also mentioned a specific method to change lr using learning rate scheduler

# Validation loop
def get_validation(model, val_dl, device):
    model.eval()
    count = 0
    corr = 0

    with torch.no_grad():
        for batch in val_dl:

            images = batch[0].to(device)
            labels = batch[1].to(device)

            count += len(labels)

            output = model(images)
            predictions = torch.argmax(output, dim=1)

            corr += (predictions == labels).sum().item()
        
        accuracy = corr/count
        print(f"correct predictions: {corr}")
        print(f"total labels: {count}")
        print(f"Accuracy: {accuracy:.2f}")
        return accuracy

# Train loop
def train_model(config):
    # Device
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(exist_ok=True)

    # get the dataset and model
    train_dataloader, test_dataloader, cutmix_or_mixup = load_dataloader(config)

    model = get_model(config).to(device)

    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])

    # optimizer and lr scheduling (Linear for first 20 then cosine annealing for rest)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.05)
    linear_scheduler = LinearLR(optimizer, start_factor=1e-3, end_factor=1 ,total_iters=20)
    decay_scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"]-20, eta_min=1e-6)
    scheduler = SequentialLR(
        optimizer,
        [linear_scheduler, decay_scheduler],
        [20]
    )
    # Extras: using the AMP to speed up training since using A100
    scaler = torch.amp.GradScaler(device=device_str)
    initial_epoch = 0
    global_step = 0
    best_accuracy = 0.0

    # preload model if training is interrupted or continue from checkpoints
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading saved model: {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        global_step = state['global_step']
        best_accuracy = state["best_acc"]

    # loss fn
    loss_fn = nn.CrossEntropyLoss().to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            images, labels = cutmix_or_mixup(images, labels)
            with torch.autocast(device_type=device_str, dtype=torch.float16):
                output = model(images)
                loss = loss_fn(output, labels)
            batch_iterator.set_postfix({f'loss': f"{loss.item():6.3f}"})

            # Tensorboard loss
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)

            # Update the weights
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            global_step += 1

        current_acc = get_validation(model, test_dataloader, device)
        # Tensorboard accuracy
        writer.add_scalar('validation_acc', current_acc, global_step)

        # Step the lr scheduler
        scheduler.step()

        # Save model every 20 epochs and also last_model to continue training if interrupted and also save best model
        if epoch%20==0:
            model_filename = get_weights_file_path(config, f'{epoch:02d}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "global_step": global_step,
                "best_acc": best_accuracy
            }, model_filename)
        # Save Best
        if current_acc > best_accuracy:
            best_accuracy = current_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "global_step": global_step,
                "best_acc":best_accuracy
            }, str(Path(config['model_folder'])/"best_model"))

        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "global_step": global_step,
                "best_acc":best_accuracy
            }, str(Path(config['model_folder'])/"last_model"))

if __name__ == "__main__":
    config = get_config()
    print(config)

    train_model(config)