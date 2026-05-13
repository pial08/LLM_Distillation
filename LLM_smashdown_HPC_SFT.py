import logging
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import json
import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from datasets import load_from_disk, concatenate_datasets


from peft import LoraConfig

import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig


def load_config(config_path):
    """Loads configuration parameters from a JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
config = load_config("config_sft.json")


class TeacherDistillationDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }

def load_teacher_model_and_tokenizer(model_name: str, device: torch.device, use_fp16: bool = True):
    """
    Loads teacher tokenizer and model for LLaMA/Mistral-style decoder-only models.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # LLaMA/Mistral models often do not define a pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if use_fp16 and torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype
    )
    model.to(device)
    model.eval()

    return model, tokenizer


#From pre-trained. model
def create_student_model(student_config: Dict, device: torch.device, use_fp16: bool = True):
    model_name = config["student_model"]["base_model_name"] 
    dtype = torch.float16 if use_fp16 and torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # LLaMA/Mistral models often do not define a pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if use_fp16 and torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype
    )
    return model.to(device), tokenizer

# def create_student_model(student_config: Dict, device: torch.device):
#     """
#     Creates a smaller LLaMA/Mistral-style student model and its tokenizer.

#     The tokenizer is created inside the function from the base model name.
#     The returned tokenizer should be used for student training/inference.
#     """
#     base_model_name = student_config["base_model_name"]

#     # Create tokenizer inside the function
#     student_tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)

#     if student_tokenizer.pad_token is None:
#         student_tokenizer.pad_token = student_tokenizer.eos_token

#     # Load base config, then shrink it
#     config = AutoConfig.from_pretrained(base_model_name)

#     if "hidden_size" in student_config:
#         config.hidden_size = student_config["hidden_size"]
#     if "intermediate_size" in student_config:
#         config.intermediate_size = student_config["intermediate_size"]
#     if "num_hidden_layers" in student_config:
#         config.num_hidden_layers = student_config["num_hidden_layers"]
#     if "num_attention_heads" in student_config:
#         config.num_attention_heads = student_config["num_attention_heads"]
#     if "num_key_value_heads" in student_config:
#         config.num_key_value_heads = student_config["num_key_value_heads"]
#     if "max_position_embeddings" in student_config:
#         config.max_position_embeddings = student_config["max_position_embeddings"]

#     # Align model config with the student tokenizer
#     config.vocab_size = len(student_tokenizer)
#     config.pad_token_id = student_tokenizer.pad_token_id
#     config.eos_token_id = student_tokenizer.eos_token_id
#     config.bos_token_id = student_tokenizer.bos_token_id

#     model = AutoModelForCausalLM.from_config(config)
#     model.to(device)

#     return model, student_tokenizer


def freeze_lower_layers(model, num_trainable_layers=4):
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze lm_head
    if hasattr(model, "lm_head"):
        for param in model.lm_head.parameters():
            param.requires_grad = True

    # Unfreeze final transformer layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        total_layers = len(model.model.layers)
        start_idx = max(0, total_layers - num_trainable_layers)

        for layer in model.model.layers[start_idx:]:
            for param in layer.parameters():
                param.requires_grad = True

    return model

def build_token_windows_from_wikitext(
    tokenizer,
    split: str = "train",
    input_token_length: int = 128,
    stride: int = 128,
    max_samples: int = None,
):
    """
    Converts WikiText into fixed token windows.

    Each sample is a fixed window of input_token_length tokens.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    text_list = [x["text"].strip() for x in dataset if x["text"] and x["text"].strip()]
    full_text = "\n\n".join(text_list)

    token_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        return_attention_mask=False
    )["input_ids"]

    windows = []
    for start in range(0, len(token_ids) - input_token_length, stride):
        window_ids = token_ids[start:start + input_token_length]
        if len(window_ids) < input_token_length:
            continue

        windows.append(window_ids)

        if max_samples is not None and len(windows) >= max_samples:
            break

    return windows


def prepare_mixed_dataset(
    max_length=256,
    total_samples=10000,
    wikitext_ratio=0.5,
    seed=42,
    split="train",
):
    """
    Load WikiText-2 and SQuAD, convert them to a common text format,
    sample a user-defined amount, combine, shuffle, tokenize, and return.

    Args:
        tokenizer: Hugging Face tokenizer
        max_length: max token length
        total_samples: total number of examples in final mixed dataset
        wikitext_ratio: fraction from WikiText; rest from SQuAD
        seed: random seed
        split: 'train' or 'validation'

    Returns:
        tokenized_dataset
    """

    print("Loading WikiText-2 and SQuAD...")

    # 1) Load datasets
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    squad = load_dataset("rajpurkar/squad", split=split)

    # 2) Convert each to a common "text" field
    # WikiText already has "text", but some rows are empty
    wikitext = wikitext.filter(lambda x: x["text"] is not None and x["text"].strip() != "")
    wikitext = wikitext.map(lambda x: {"text": x["text"]})

    # Convert SQuAD row into a single text string
    def squad_to_text(example):
        answer_text = ""
        if example["answers"]["text"]:
            answer_text = example["answers"]["text"][0]

        combined = (
            f"Context: {example['context']}\n"
            f"Question: {example['question']}\n"
            f"Answer: {answer_text}"
        )
        return {"text": combined}

    squad = squad.map(squad_to_text)

    # Keep only the text column
    wikitext = wikitext.remove_columns([c for c in wikitext.column_names if c != "text"])
    squad = squad.remove_columns([c for c in squad.column_names if c != "text"])

    # 3) Shuffle before sampling
    wikitext = wikitext.shuffle(seed=seed)
    squad = squad.shuffle(seed=seed)

    # 4) Decide sample counts
    num_wikitext = int(total_samples * wikitext_ratio)
    num_squad = total_samples - num_wikitext

    num_wikitext = min(num_wikitext, len(wikitext))
    num_squad = min(num_squad, len(squad))

    wikitext = wikitext.select(range(num_wikitext))
    squad = squad.select(range(num_squad))

    print(f"Selected {len(wikitext)} WikiText samples")
    print(f"Selected {len(squad)} SQuAD samples")

    # 5) Combine and reshuffle
    combined_dataset = concatenate_datasets([wikitext, squad]).shuffle(seed=seed)
    return combined_dataset
    # 6) Tokenize
    def tokenize_function(examples):
        texts = [text + tokenizer.eos_token for text in examples["text"]]
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        # return tokenizer(
        #     #examples["text"],
        #     truncation=True,
        #     padding="max_length",
        #     max_length=max_length,
        # )

    tokenized_dataset = combined.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return tokenized_dataset

@torch.no_grad()
def generate_teacher_outputs_for_sft(
    teacher_model,
    teacher_tokenizer,
    device: torch.device,
    input_token_length: int = 128,
    output_max_new_tokens: int = 64,
    source_split: str = "train",
    source_stride: int = 128,
    max_samples: int = 1000,
    batch_size: int = 2,
):
    """
    Generates teacher outputs for SFTTrainer.

    Output format:
    returns a Hugging Face Dataset with columns:
      - prompt
      - completion
      - prompt_text
      - target_text
      - text

    The 'prompt' and 'completion' columns are the most useful for SFTTrainer.
    """
    teacher_model.eval()

    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    token_windows = build_token_windows_from_wikitext(
        tokenizer=teacher_tokenizer,
        split=source_split,
        input_token_length=input_token_length,
        stride=source_stride,
        max_samples=max_samples,
    )

    rows = []

    for start_idx in range(0, len(token_windows), batch_size):
        batch_windows = token_windows[start_idx:start_idx + batch_size]

        batch_input_ids = torch.tensor(batch_windows, dtype=torch.long, device=device)
        batch_attention_mask = torch.ones_like(batch_input_ids, device=device)

        generated_ids = teacher_model.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            max_new_tokens=output_max_new_tokens,
            do_sample=False,
            pad_token_id=teacher_tokenizer.pad_token_id,
            eos_token_id=teacher_tokenizer.eos_token_id,
        )

        for i, prompt_ids in enumerate(batch_windows):
            full_generated_ids = generated_ids[i].tolist()
            continuation_ids = full_generated_ids[len(prompt_ids):]

            if len(continuation_ids) == 0:
                continue

            prompt_text = teacher_tokenizer.decode(
                prompt_ids,
                skip_special_tokens=True
            ).strip()

            target_text = teacher_tokenizer.decode(
                continuation_ids,
                skip_special_tokens=True
            ).strip()

            if not prompt_text or not target_text:
                continue

            prompt = f"### Input:\n{prompt_text}\n\n### Output:\n"
            completion = target_text
            text = prompt + completion

            rows.append({
                "prompt": prompt,
                "completion": completion,
                "prompt_text": prompt_text,
                "target_text": target_text,
                "text": text,
            })

        print(f"Teacher generation done for {min(start_idx + batch_size, len(token_windows))}/{len(token_windows)} samples")

    return Dataset.from_list(rows)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# teacher_model, teacher_tokenizer = load_teacher_model_and_tokenizer(
#     model_name=config["teacher_model_name"],
#     device=device,
#     use_fp16=config.get("use_fp16", True)
# )

# Step 4: create and train student
student_model, student_tokenizer = create_student_model(
    student_config=config["student_model"],
    device=device,
    use_fp16=True
    )
####    ***** Freezing Some Layers ***** ####
student_model = freeze_lower_layers(student_model, num_trainable_layers=16)



sft_config = SFTConfig(
    output_dir=config.get("result_dir"),
    per_device_train_batch_size=config.get("per_device_train_batch_size"),
    per_device_eval_batch_size=config.get("per_device_eval_batch_size"),
    learning_rate=config.get("learning_rate"),
    num_train_epochs=config.get("num_epochs"),
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    max_length=config.get("output_max_new_tokens"),
    gradient_accumulation_steps=4,
    max_grad_norm=config.get("max_grad_norm"),
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=torch.cuda.is_available(),
    fp16=False,
    report_to="none",
)



# Loading Local Dataset
# dataset = load_from_disk("Datasets/distill_data/train")
# splits = dataset.train_test_split(test_size=0.2, seed=42)

# train_dataset = splits["train"]
# temp_dataset = splits["test"]
# val_test = temp_dataset.train_test_split(test_size=0.5, seed=42)
# val_dataset = val_test["train"]
# test_dataset = val_test["test"]

# Original student model: meta-llama/Llama-3.2-1B-Instruct

# Load mixed dataset
max_length = config.get("max_length", 128)
num_samples = config.get("num_samples", 1000)
dataset = prepare_mixed_dataset(
            max_length=256,
            total_samples=num_samples,
            wikitext_ratio=0.9,
            seed=42,
            split="train",
        )

splits = dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = splits["train"]
temp_dataset = splits["test"]
val_test = temp_dataset.train_test_split(test_size=0.5, seed=42)
val_dataset = val_test["train"]
test_dataset = val_test["test"]

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=student_model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=student_tokenizer,
)



trainer.train()


# eval_metrics = trainer.evaluate(test_dataset)
# print("Test metrics:", eval_metrics)

trainer.save_model(os.path.join(sft_config.output_dir, "TestLLaMa-v-SFT"))
student_tokenizer.save_pretrained(os.path.join(sft_config.output_dir, "TestLLaMa-v-SFT"))

