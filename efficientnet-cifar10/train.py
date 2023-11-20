"""Train a model on the CIFAR10 dataset."""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor


def train(args):
    """Train a model on MNIST dataset."""
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
    else:
        local_rank = 0

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        wandb.login(
            key="local-0b4dd77e45ad93ff68db22067d0d0f3ef9323636",
            host="http://115.27.161.208:8081/",
        )
        run = wandb.init(
            project="yousiki-cifar10",
            entity="adminadmin",
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
            },
        )

    # Load the dataset
    transform = Compose(
        [
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = CIFAR10(
        root="./data", train=True, transform=transform, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_dataset = CIFAR10(
        root="./data", train=False, transform=transform, download=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Load the model
    model = models.efficientnet_b0(num_classes=10)

    # Checkpoint
    checkpoint_path = Path("app/output/checkpoint/")
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoints = list(filter(lambda p: p.is_file(), checkpoint_path.rglob("*.pth")))
    if len(checkpoints) > 0:
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        checkpoint = torch.load(checkpoints[-1])
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 1

    model.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def wandb_log(*args, **kwargs):
        if local_rank == 0:
            wandb.log(*args, **kwargs)

    def train_epoch(epoch):
        print(f"Epoch {epoch}: start training...")
        model.train()
        train_loss_epoch = 0
        for i, (images, target) in enumerate(train_loader):
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = torch.nn.functional.cross_entropy(output, target)
            train_loss_epoch += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb_log(
                {
                    "iteration": epoch * len(train_loader) + i,
                    "trained_samples": (i + 1) * args.batch_size,
                    "total_samples": len(train_dataset),
                    "train_loss_step": loss.item(),
                }
            )
        wandb_log({"train_loss_epoch": train_loss_epoch / len(train_loader)})
        print(f"Epoch {epoch}: training loss: {train_loss_epoch / len(train_loader)}")

    @torch.inference_mode()
    def val_epoch(epoch):
        print(f"Epoch {epoch}: start validation...")
        model.eval()
        val_loss_epoch = 0
        val_correct_epoch = 0
        for images, target in val_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = torch.nn.functional.cross_entropy(output, target)
            val_loss_epoch += loss.item()

            pred = output.argmax(dim=1)
            val_correct_epoch += pred.eq(target).sum().item()
        wandb_log(
            {
                "val_loss_epoch": val_loss_epoch / len(val_loader),
                "val_acc_epoch": val_correct_epoch / len(val_dataset),
            }
        )
        print(f"Epoch {epoch}: validation loss: {val_loss_epoch / len(val_loader)}")
        print(f"Epoch {epoch}: validation acc: {val_correct_epoch / len(val_dataset)}")

    def save_checkpoint(epoch):
        if local_rank == 0:
            state_dict = model.state_dict()
            if hasattr(model, "module"):
                state_dict = model.module.state_dict()
            checkpoint = {
                "model": state_dict,
                "epoch": epoch,
            }
            torch.save(checkpoint, checkpoint_path / f"{epoch}/{epoch:0d}.pth")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        train_epoch(epoch)
        val_epoch(epoch)
        save_checkpoint(epoch)

    if local_rank == 0:
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    args = parser.parse_args()

    train(args)
