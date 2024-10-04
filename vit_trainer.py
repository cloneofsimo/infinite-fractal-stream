import torch
import torch.nn as nn

import click
from transformers import get_cosine_schedule_with_warmup
import wandb
from fractal_dataset import FractalImageDataset
from model import create_vit_model

import numpy as np
import random

# from zeroshampoo import Shampoo


def get_optimizer(optimizer_type, model, lr_embedding, lr_output, lr_rest):
    embedding_params = list(model.patch_embed.parameters())
    output_params = list(model.head.parameters())
    rest_params = []
    for name, param in model.named_parameters():
        if not any([name.startswith("patch_embed"), name.startswith("head")]):
            rest_params.append(param)

    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(
            [
                {"params": embedding_params, "lr": lr_embedding},
                {"params": output_params, "lr": lr_rest},
                {"params": rest_params, "lr": lr_rest},
            ]
        )
    elif optimizer_type.lower() == "shampoo":
        optimizer = Shampoo(
            [
                {"params": embedding_params, "lr": lr_embedding},
                {"params": output_params, "lr": lr_rest},
                {"params": rest_params, "lr": lr_rest},
            ]
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    return optimizer


def evaluate(
    model, dataset, criterion, device, eval_steps, logger, batch_size, at_step=None
):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    dataset.reset()
    with torch.no_grad():
        for _ in range(eval_steps):
            images, labels = dataset.get_batch(batch_size)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    print(f"Eval Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    logger.log(
        {
            "eval_loss": avg_loss,
            "eval_accuracy": accuracy,
            f"eval_loss/loss_{at_step}": avg_loss,
            f"eval_accuracy/acc_{at_step}": accuracy,
        }
    )
    model.train()


@click.command()
@click.option("--batch_size", default=32, help="Batch size for training")
@click.option("--num_steps", default=1000, help="Number of training steps")
@click.option("--eval_steps", default=100, help="Number of steps between evaluations")
@click.option("--num_classes", default=10, help="Number of classes")
@click.option("--image_size", default=128, help="Size of the generated images")
@click.option(
    "--max_iter", default=30, help="Maximum iterations for fractal generation"
)
@click.option("--embed_dim", default=256, help="Embedding dimension for ViT")
@click.option("--depth", default=12, help="Depth of the transformer")
@click.option("--num_heads", default=8, help="Number of heads in ViT")
@click.option(
    "--lr_embedding", default=1e-4, help="Learning rate for embedding parameters"
)
@click.option("--lr_output", default=1e-4, help="Learning rate for output parameters")
@click.option(
    "--lr_rest", default=1e-4, help="Learning rate for the rest of the parameters"
)
@click.option(
    "--optimizer_type", default="adam", help="Optimizer type: adam or shampoo"
)
@click.option(
    "--warmup_steps", default=1000, help="Number of warmup steps for the scheduler"
)
@click.option("--device", default="cuda", help="Device to use for training")
@click.option("--run_name", default="fractal_vit_training", help="WandB run name")
@click.option("--seed", default=0, help="Seed for the random number generator")
@click.option(
    "--num_samples_per_class", default=1000, help="Number of samples per class"
)
@click.option("--eval_once_every", default=100, help="Evaluate once every X steps")
def main(
    batch_size,
    num_steps,
    eval_steps,
    num_classes,
    image_size,
    max_iter,
    embed_dim,
    depth,
    num_heads,
    lr_embedding,
    lr_output,
    lr_rest,
    optimizer_type,
    warmup_steps,
    device,
    run_name,
    seed,
    num_samples_per_class,
    eval_once_every,
):

    # seed everything
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    wandb.init(
        project="Fractal_ViT_Classifier",
        name=run_name,
        entity="simo",
        config={
            "batch_size": batch_size,
            "num_steps": num_steps,
            "eval_steps": eval_steps,
            "num_classes": num_classes,
            "image_size": image_size,
            "max_iter": max_iter,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "lr_embedding": lr_embedding,
            "lr_output": lr_output,
            "lr_rest": lr_rest,
            "optimizer_type": optimizer_type,
            "warmup_steps": warmup_steps,
        },
    )
    logger = wandb

    model = create_vit_model(
        image_size=image_size,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
    )
    model.to(device)

    optimizer = get_optimizer(optimizer_type, model, lr_embedding, lr_output, lr_rest)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps
    )

    criterion = nn.CrossEntropyLoss()

    train_dataset = FractalImageDataset(
        num_classes=num_classes,
        num_samples_per_class=num_samples_per_class,
        image_size=image_size,
        max_iter=max_iter,
        R=2,
        device=device,
        train=True,
        seed=seed,
    )

    eval_dataset = FractalImageDataset(
        num_classes=num_classes,
        num_samples_per_class=1000,
        image_size=image_size,
        max_iter=max_iter,
        R=2,
        device=device,
        train=False,
        seed=seed,
    )

    step = 0
    while step < num_steps:
        images, labels = train_dataset.get_batch(batch_size)
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        logger.log({"train_loss": loss.item(), "step": step})

        if step % eval_once_every == 0:
            evaluate(
                model,
                eval_dataset,
                criterion,
                device,
                eval_steps,
                logger,
                batch_size,
                step,
            )

        step += 1

    wandb.finish()

    return 0


if __name__ == "__main__":
    main()
