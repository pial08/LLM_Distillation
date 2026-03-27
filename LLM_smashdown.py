#!/usr/bin/env python
"""
Smart Knowledge Distillation Script with Mixed Precision, Optimized Data Loading,
and Saving of Trained Student Model Parameters

This script trains student models (with varying transformer block numbers) using knowledge distillation
from a fixed teacher model. For each student (with layers ranging from 4 to 8), the training runs
for a specified number of epochs on the wikitext-2-raw-v1 dataset and:
  - Saves per-step losses (both distillation and standard cross-entropy losses) into a CSV file
    named by the student layer.
  - Saves the trained model parameters (state_dict) into a .pt file for future use.
After training all models, the script generates and saves plots for accumulated losses and average epoch training times.

Configuration is provided via a "config.json" file in the same directory.
Sample config.json:
{
    "result_dir": "results",
    "teacher_model_name": "distilgpt2",
    "max_length": 128,
    "batch_size": 32,
    "num_epochs": 100,
    "learning_rate": 0.0001,
    "temperature": 2.0,
    "student_layer_range": [4, 8],
    "num_workers": 4
}

Usage:
    python LLM_smashdown.py

Note on Runtime (rough estimate):
- Our previous log showed a 1-layer student model took about 2 minutes per epoch.
- Expecting roughly linear scaling, a 4-8 layer student model may take around 8-16 minutes per epoch.
- For 100 epochs, that is roughly 800-1600 minutes per configuration (about 13-27 hours),
  so you may need to reduce epochs if total runtime is a concern.
"""

import os
import time
import json
import logging
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

# Import AMP utilities for mixed precision training
from torch.cuda.amp import autocast, GradScaler


def setup_logging(result_dir):
    """Configures logging to output both to file and console."""
    log_file = os.path.join(result_dir, "training_log.txt")
    logging.basicConfig(level=logging.INFO,
                        filename=log_file,
                        filemode="w",
                        format="%(asctime)s - %(levelname)s - %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def load_config(config_path):
    """Loads configuration parameters from a JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def prepare_dataset(tokenizer, max_length):
    """Loads and tokenizes the wikitext dataset."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized_dataset


def create_student_model(teacher_model, tokenizer, student_layers):
    """
    Initializes a student model with a given number of transformer blocks.
    Copies the token embeddings and final linear layer weights from the teacher,
    then freezes these layers.
    """
    teacher_config = teacher_model.config
    student_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=128,
        n_ctx=128,
        n_embd=teacher_config.n_embd,       # Match teacher hidden size
        n_layer=student_layers,             # Set number of student layers
        n_head=teacher_config.n_head        # Use same head count as teacher
    )
    student_model = GPT2LMHeadModel(student_config)
    with torch.no_grad():
        # Copy token embeddings and final linear layer weights
        student_model.transformer.wte.weight.copy_(teacher_model.transformer.wte.weight)
        # Optionally copy positional embeddings if desired:
        # student_model.transformer.wpe.weight.copy_(teacher_model.transformer.wpe.weight)
        student_model.lm_head.weight.copy_(teacher_model.lm_head.weight)
    # Freeze the embedding and final linear layer parameters
    for param in student_model.transformer.wte.parameters():
        param.requires_grad = False
    for param in student_model.lm_head.parameters():
        param.requires_grad = False
    return student_model


def train_student_model(student_model, teacher_model, dataloader, device, optimizer, num_epochs, temperature, tokenizer):
    """
    Trains the student model using knowledge distillation with mixed precision.
    Tracks per-step distillation loss (KL divergence) and standard language-modeling (cross-entropy) loss.
    Returns training loss history and per-epoch training times.
    """
    steps = []
    distill_losses = []
    ce_losses = []
    epoch_times = []
    student_model.train()
    global_step = 0

    # Initialize GradScaler for mixed precision training.
    scaler = GradScaler()

    for epoch in range(num_epochs):
        start_time = time.time()
        total_distill = 0.0
        total_ce = 0.0
        step_count = 0
        
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast():
                # Teacher forward pass (no gradient tracking)
                with torch.no_grad():
                    teacher_outputs = teacher_model(**batch)
                teacher_logits = teacher_outputs.logits

                # Student forward pass
                student_outputs = student_model(**batch)
                student_logits = student_outputs.logits

                # Calculate distillation loss with temperature scaling
                teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
                student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
                loss_distill = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

                # Calculate standard language modeling loss (cross-entropy)
                loss_ce = F.cross_entropy(
                    student_logits.view(-1, student_logits.size(-1)),
                    batch["input_ids"].view(-1),
                    ignore_index=tokenizer.pad_token_id
                )

                # Here we use only the distillation loss for backpropagation.
                loss = loss_distill

            # Scale the loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_distill += loss_distill.item()
            total_ce += loss_ce.item()
            global_step += 1
            steps.append(global_step)
            distill_losses.append(loss_distill.item())
            ce_losses.append(loss_ce.item())
            step_count += 1

            if global_step % 100 == 0:
                logging.info(f"Epoch {epoch+1}/{num_epochs}, Global Step {global_step}, Distill Loss: {loss_distill.item():.4f}, CE Loss: {loss_ce.item():.4f}")

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        avg_distill = total_distill / step_count if step_count > 0 else 0
        avg_ce = total_ce / step_count if step_count > 0 else 0
        logging.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f} sec | Avg Distill Loss: {avg_distill:.4f} | Avg CE Loss: {avg_ce:.4f}")

    return {"steps": steps, "distill_losses": distill_losses, "ce_losses": ce_losses, "epoch_times": epoch_times}


def save_csv(loss_dict, csv_path):
    """Saves step-wise loss data (and cumulative losses) to a CSV file."""
    df = pd.DataFrame({
        "Step": loss_dict["steps"],
        "Distillation Loss": loss_dict["distill_losses"],
        "CE Loss": loss_dict["ce_losses"]
    })
    df["Accumulated Distillation Loss"] = df["Distillation Loss"].cumsum()
    df["Accumulated CE Loss"] = df["CE Loss"].cumsum()
    df.to_csv(csv_path, index=False)


def plot_results(result_dir, results_dict, epoch_time_dict):
    """Generates and saves plots for accumulated losses and average epoch training time."""
    # Plot accumulated distillation loss
    plt.figure(figsize=(10, 6))
    for layer, loss_dict in results_dict.items():
        df = pd.DataFrame({
            "Step": loss_dict["steps"],
            "Accumulated Distillation Loss": pd.Series(loss_dict["distill_losses"]).cumsum()
        })
        plt.plot(df["Step"], df["Accumulated Distillation Loss"], label=f"Student Layer {layer}")
    plt.xlabel("Step")
    plt.ylabel("Accumulated Distillation Loss")
    plt.title("Accumulated Distillation Loss vs Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "accumulated_distillation_loss.jpg"))
    plt.close()

    # Plot accumulated cross-entropy loss
    plt.figure(figsize=(10, 6))
    for layer, loss_dict in results_dict.items():
        df = pd.DataFrame({
            "Step": loss_dict["steps"],
            "Accumulated CE Loss": pd.Series(loss_dict["ce_losses"]).cumsum()
        })
        plt.plot(df["Step"], df["Accumulated CE Loss"], label=f"Student Layer {layer}")
    plt.xlabel("Step")
    plt.ylabel("Accumulated CE Loss")
    plt.title("Accumulated Cross-Entropy Loss vs Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "accumulated_ce_loss.jpg"))
    plt.close()

    # Plot average epoch training time for each student model
    layers = sorted(epoch_time_dict.keys())
    avg_times = [sum(epoch_time_dict[l]) / len(epoch_time_dict[l]) for l in layers]
    plt.figure(figsize=(10, 6))
    plt.bar([str(l) for l in layers], avg_times)
    plt.xlabel("Student Layer Number")
    plt.ylabel("Avg Epoch Training Time (sec)")
    plt.title("Average Training Time per Epoch by Student Layer")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "epoch_training_time.jpg"))
    plt.close()


def main():
    # Load configuration from config.json
    config_path = "config.json"
    config = load_config(config_path)
    result_dir = config.get("result_dir", "results")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    setup_logging(result_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Starting training on GPU" if torch.cuda.is_available() else "Starting training on CPU")
    
    # Load teacher model and tokenizer
    teacher_model_name = config.get("teacher_model_name", "distilgpt2")
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher_model.eval()
    logging.info(f"Teacher model: {teacher_model_name} | Transformer blocks: {teacher_model.config.n_layer}")
    
    # Prepare dataset and DataLoader with optimized data loading
    max_length = config.get("max_length", 128)
    tokenized_dataset = prepare_dataset(tokenizer, max_length)
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get training hyperparameters from config
    num_epochs = config.get("num_epochs", 100)
    learning_rate = config.get("learning_rate", 1e-4)
    temperature = config.get("temperature", 2.0)
    # Note: Update the range to only test student models with 4–8 layers
    student_layer_range = config.get("student_layer_range", [4, 8])
    
    results_dict = {}      # To store loss histories for each student configuration
    epoch_time_dict = {}   # To store per-epoch training time for each configuration
    
    # Loop over student models for each configuration in the given range
    for student_layers in range(student_layer_range[0], student_layer_range[1] + 1):
        logging.info(f"----- Training Student Model with {student_layers} Transformer Blocks -----")
        student_model = create_student_model(teacher_model, tokenizer, student_layers).to(device)
        optimizer = Adam(student_model.parameters(), lr=learning_rate)
        
        # Train the student model
        loss_dict = train_student_model(student_model, teacher_model, dataloader, device, optimizer, num_epochs, temperature, tokenizer)
        results_dict[student_layers] = loss_dict
        epoch_time_dict[student_layers] = loss_dict["epoch_times"]
        
        # Save step-wise losses to CSV named by student layer
        csv_filename = os.path.join(result_dir, f"student_layer_{student_layers}.csv")
        save_csv(loss_dict, csv_filename)
        logging.info(f"Finished training student model with {student_layers} layers. CSV saved to {csv_filename}.")
        
        # Save the trained model parameters for future use
        model_filename = os.path.join(result_dir, f"student_model_{student_layers}.pt")
        torch.save(student_model.state_dict(), model_filename)
        logging.info(f"Saved trained student model with {student_layers} layers to {model_filename}.")
    
    # Generate and save plots for losses and training times
    logging.info("Generating plots for accumulated losses and training times.")
    try:
        plot_results(result_dir, results_dict, epoch_time_dict)
        logging.info("Plots generated successfully.")
    except Exception as e:
        logging.error(f"Error generating plots: {e}")
    
    logging.info("All experiments completed successfully.")


if __name__ == "__main__":
    main()
