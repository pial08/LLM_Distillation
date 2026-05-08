#!/usr/bin/env python
"""
Smart Knowledge Distillation Script for Supercomputer Cluster

This script trains student models (with varying transformer block numbers) using knowledge distillation
from a fixed teacher model. For each student (with layers ranging from 1 to 10), the training runs
for 200 epochs on the wikitext-2-raw-v1 dataset and saves step-wise losses (both distillation and standard
cross-entropy losses) into a CSV file named by the student layer. After training all models, the script
generates and saves plots for accumulated losses and average epoch training times.

Configuration is provided via a "config.json" file (see sample config below).

Logging is written to a log file in the results directory, ensuring that all progress is saved when running
via the cluster scheduler.

Sample config.json:
{
    "result_dir": "results",
    "teacher_model_name": "distilgpt2",
    "max_length": 128,
    "batch_size": 8,
    "num_epochs": 200,
    "learning_rate": 0.0001,
    "temperature": 2.0,
    "student_layer_range": [1, 10]
}

Usage:
    python distillation_cluster.py

Make sure that the "config.json" file is in the same directory as this script.
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPT2LMHeadModel
from transformers import LlamaConfig, LlamaForCausalLM
from datasets import load_from_disk



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


def get_torch_dtype(dtype_name: str | None):
    if dtype_name is None:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype in config: {dtype_name}")
    return mapping[key]

def build_bnb_config(qcfg: dict | None):
    if not qcfg or not qcfg.get("enabled", False):
        return None

    mode = qcfg.get("mode", "").lower()

    if mode == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=qcfg.get("llm_int8_threshold", 6.0),
            llm_int8_enable_fp32_cpu_offload=qcfg.get("llm_int8_enable_fp32_cpu_offload", False),
            llm_int8_skip_modules=qcfg.get("llm_int8_skip_modules"),
        )

    if mode == "4bit":
        compute_dtype = get_torch_dtype(qcfg.get("bnb_4bit_compute_dtype", "bfloat16"))
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=qcfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    raise ValueError(f"Unsupported quantization mode: {mode}")


def load_config(config_path):
    """Loads configuration parameters from a JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def prepare_dataset_HF(tokenizer, max_length):
    print("Using HF Dataset")
    """Loads and tokenizes the wikitext dataset."""
    #dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train[:5000]"
    )
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized_dataset


def prepare_dataset_local(tokenizer, max_length):
    print("Using HF Dataset")
    """Loads dataset from disk, splits it, and tokenizes train/val/test."""
    print("Inside dataset loader")
    TEST_DATASET_PATH = "Datasets/distill_data/train"
    dataset = load_from_disk(TEST_DATASET_PATH)

    # 80% train, 10% validation, 10% test
    splits = dataset.train_test_split(test_size=0.06, seed=42)

    train_dataset = splits["train"]
    # temp_dataset = splits["test"]

    # print("Printing training dataset", train_dataset)
    # val_test = temp_dataset.train_test_split(test_size=0.5, seed=42)

    # val_dataset = val_test["train"]
    # test_dataset = val_test["test"]
    print("Printing a sample dataset", train_dataset[0])
    print("----Dataset Preparation----")    
    def tokenize_function(examples):
        texts = [
            f"{prompt}\n{completion}"
            for prompt, completion in zip(examples["prompt"], examples["completion"])
        ]
        

        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    # val_dataset = val_dataset.map(tokenize_function, batched=True)
    # test_dataset = test_dataset.map(tokenize_function, batched=True)
    #print("length of train dataset ", len(train_dataset))
    columns = ["input_ids", "attention_mask"]

    train_dataset.set_format(type="torch", columns=columns)
    # val_dataset.set_format(type="torch", columns=columns)
    # test_dataset.set_format(type="torch", columns=columns)

    #return train_dataset, val_dataset, test_dataset
    return train_dataset

def create_teacher_model(config: dict):
    teacher_model_name = config.get("teacher_model_name", "distilgpt2")
    if config.get("quantization"):
        print("Quantization is True")
        if config.get("quantizatio_bit") == 4:
            print("4 Bit Quantization")
            qcfg = config.get("teacher_quantization_4", {})
        else:
            print("8 Bit Quantization")
            qcfg = config.get("teacher_quantization_8", {})

        tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = build_bnb_config(qcfg)
        model_kwargs = {
            "pretrained_model_name_or_path": teacher_model_name,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = qcfg.get("device_map", "auto")
            dtype = qcfg.get("torch_dtype", "auto")
            model_kwargs["dtype"] = dtype  # transformers docs support dtype="auto"
        else:
            torch_dtype = get_torch_dtype(config.get("teacher_torch_dtype"))
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype

        teacher_model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        teacher_model.eval()
    else:
        print("Quantization is false")
        # Will be inside an else
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name).to(qcfg.get("device_map", "auto"))
        tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        teacher_model.eval()

    

    return teacher_model, tokenizer

def create_student_model(teacher_model, tokenizer, student_layers: int, max_positions: int = 2048):
    """
    Initializes a LLaMA-style student model with a given number of transformer blocks.
    Copies the token embeddings and final linear layer weights from the teacher,
    then freezes these layers.
    """
    teacher_config = teacher_model.config


    # Build a LLaMA config that matches the teacher's width/heads, but fewer layers
    student_config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=teacher_config.hidden_size,
        intermediate_size=teacher_config.intermediate_size, # experiment by changing the intermediate size
        num_hidden_layers=student_layers,
        num_attention_heads=teacher_config.num_attention_heads,
        num_key_value_heads=getattr(teacher_config, "num_key_value_heads", teacher_config.num_attention_heads),
        max_position_embeddings=getattr(teacher_config, "max_position_embeddings", max_positions),
        rms_norm_eps=getattr(teacher_config, "rms_norm_eps", 1e-5),
        rope_theta=getattr(teacher_config, "rope_theta", 10000.0),
        rope_scaling=getattr(teacher_config, "rope_scaling", None),
        bos_token_id=teacher_config.bos_token_id,
        eos_token_id=teacher_config.eos_token_id,
        pad_token_id=getattr(teacher_config, "pad_token_id", tokenizer.pad_token_id),
    )

    student_model = LlamaForCausalLM(student_config)

    print(teacher_model)

    n_params = sum(p.numel() for p in teacher_model.parameters())
    print("Teacher model parameters --> billions:", n_params / 1e9)


    print("-----------------------------------------------")
    print(student_model)
    n_params = sum(p.numel() for p in student_model.parameters())
    print("Student model parameters --> billions:", n_params / 1e9)

    with torch.no_grad():
        # Copy token embeddings (LLaMA: model.embed_tokens)
        student_model.model.embed_tokens.weight.copy_(
            teacher_model.model.embed_tokens.weight
        )

        # Copy final linear layer (lm_head)
        student_model.lm_head.weight.copy_(teacher_model.lm_head.weight)
        if getattr(student_model.lm_head, "bias", None) is not None and getattr(teacher_model.lm_head, "bias", None) is not None:
            student_model.lm_head.bias.copy_(teacher_model.lm_head.bias)

    # Freeze the embedding and final linear layer
    for param in student_model.model.embed_tokens.parameters():
        param.requires_grad = False
    for param in student_model.lm_head.parameters():
        param.requires_grad = False

    return student_model

def train_student_model(student_model, teacher_model, dataloader, device, optimizer, num_epochs, temperature, tokenizer):
    """
    Trains the student model using knowledge distillation.
    Tracks per-step distillation loss (KL divergence) and standard language-modeling (cross-entropy) loss.
    Also records epoch training times.
    """
    steps = []
    distill_losses = []
    ce_losses = []
    epoch_times = []
    student_model.train()
    global_step = 0
    for epoch in range(num_epochs):
        start_time = time.time()
        total_distill = 0.0
        total_ce = 0.0
        step_count = 0
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            # Teacher forward pass (no gradient update)
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch)
            teacher_logits = teacher_outputs.logits
            # Student forward pass
            student_outputs = student_model(**batch)
            student_logits = student_outputs.logits
            # Compute distillation loss using temperature scaling
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            loss_distill = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
            # Compute standard LM loss (cross-entropy)
            loss_ce = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                batch["input_ids"].view(-1),
                ignore_index=tokenizer.pad_token_id
            )
            # Backpropagate using the distillation loss (you can combine losses if desired)
            loss = loss_distill
            loss.backward()
            optimizer.step()
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
    """Saves the step-wise loss data (and cumulative losses) into a CSV file."""
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

    # Plot accumulated standard LM (CE) loss
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
    # Load configuration
    config_path = "config.json"
    config = load_config(config_path)
    result_dir = config.get("result_dir", "results")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    setup_logging(result_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Starting training on GPU" if torch.cuda.is_available() else "Starting training on CPU")
    
    # Load teacher model and tokenizer (original) -- commenting temp
    teacher_model_name = config.get("teacher_model_name", "distilgpt2")
    # teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name).to(device)
    # tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    # teacher_model.eval()

    teacher_model, tokenizer = create_teacher_model(config)





    # For GPT Style Models
    #logging.info(f"Teacher model: {teacher_model_name} | Transformer blocks: {teacher_model.config.n_layer}")
    # for LLaMa Style Models
    logging.info(f"Teacher model: {teacher_model_name} | Transformer blocks: {teacher_model.config.num_hidden_layers}")

    # Prepare dataset and dataloader (same for all experiments)
    max_length = config.get("max_length", 128)
    
    if config.get("local_dataset"):
        tokenized_dataset = prepare_dataset_local(tokenizer, max_length)
    else:
        tokenized_dataset = prepare_dataset_HF(tokenizer, max_length)
    
    batch_size = config.get("batch_size", 8)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
    
    # Get training parameters from config
    num_epochs = config.get("num_epochs", 200)
    learning_rate = config.get("learning_rate", 1e-4)
    temperature = config.get("temperature", 2.0)
    student_layer_range = config.get("student_layer_range", [1, 10])
    
    results_dict = {}      # To store loss information for each student configuration
    epoch_time_dict = {}   # To store per-epoch training time
    
    # Loop over student models with transformer block numbers from student_layer_range[0] to student_layer_range[1]
    for student_layers in range(student_layer_range[0], student_layer_range[1] + 1):
        logging.info(f"----- Training Student Model with {student_layers} Transformer Blocks -----")
        student_model = create_student_model(teacher_model, tokenizer, student_layers).to(device)
        optimizer = Adam(student_model.parameters(), lr=learning_rate)
        loss_dict = train_student_model(student_model, teacher_model, dataloader, device, optimizer, num_epochs, temperature, tokenizer)
        results_dict[student_layers] = loss_dict
        epoch_time_dict[student_layers] = loss_dict["epoch_times"]
        # Save step-wise loss to CSV (file will be named by student layer)
        csv_filename = os.path.join(result_dir, f"student_layer_{student_layers}.csv")
        save_csv(loss_dict, csv_filename)
        logging.info(f"Finished training student model with {student_layers} layers. CSV saved to {csv_filename}.")
        
        # Saving the model
        #save_dir = "saved_models/student_model_" + "student_layers"
        save_dir = "saved_models/TestLLaMa-v1.0"
        student_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    
    # Generate and save plots
    logging.info("Generating plots for accumulated losses and training times.")
    try:
        plot_results(result_dir, results_dict, epoch_time_dict)
        logging.info("Plots generated successfully.")
    except Exception as e:
        logging.error(f"Error generating plots: {e}")
    
    logging.info("All experiments completed successfully.")

if __name__ == "__main__":
    main()
