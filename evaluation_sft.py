import os
import json
import math
from typing import Dict, List

import torch
import evaluate
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F


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



def load_trained_model(model_dir: str, device: torch.device):
    """
    Load a trained causal LM and tokenizer from a local directory.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype
    ).to(device)
    model.eval()

    return model, tokenizer



def load_test_dataset(test_dataset_path: str):
    """
    Load a Hugging Face dataset split saved with save_to_disk().
    Expected columns:
      - text
      - prompt_text
      - target_text
    """

    # Returning test dataset only
    dataset = load_from_disk("Datasets/distill_data/train")
    splits = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = splits["train"]
    temp_dataset = splits["test"]
    val_test = temp_dataset.train_test_split(test_size=0.5, seed=42)
    val_dataset = val_test["train"]
    test_dataset = val_test["test"]
    return test_dataset

    # dataset = load_from_disk(test_dataset_path)
    # return dataset


def build_prompt(prompt_text: str) -> str:
    """
    Rebuild the generation prompt from prompt_text.
    Must match the format used during training data creation.
    """
    return f"### Input:\n{prompt_text.strip()}\n\n### Output:\n"


@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 256,
):
    """
    Generate only the continuation after the prompt.
    """
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    output_ids = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_len = enc["input_ids"].shape[1]
    gen_ids = output_ids[0][prompt_len:]
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return generated_text

def compute_cosine_similarity(
    embedder,
    predictions: List[str],
    references: List[str],
    device: str = None,
) -> List[float]:
    """
    Compute pairwise cosine similarity using sentence embeddings.
    """
    pred_emb = embedder.encode(
        predictions,
        convert_to_tensor=True,
        show_progress_bar=False,
        device=device,
        normalize_embeddings=True,
    )
    ref_emb = embedder.encode(
        references,
        convert_to_tensor=True,
        show_progress_bar=False,
        device=device,
        normalize_embeddings=True,
    )

    cosine_scores = F.cosine_similarity(pred_emb, ref_emb).cpu().tolist()
    return cosine_scores


def evaluate_model(
    model_dir: str,
    test_dataset_path: str,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_new_tokens: int = 256,
    save_predictions_path: str = None,
):
    """
    Full evaluation pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load model
    model, tokenizer = load_trained_model(model_dir, device)

    # 2) Load test dataset
    #Original Test Dataset
    test_dataset = load_test_dataset(test_dataset_path)

    ### (Split training)

    # dataset = load_from_disk("Datasets/distill_data/train")

    # splits = dataset.train_test_split(test_size=0.2, seed=42)
    # train_dataset = splits["train"]
    # temp_dataset = splits["test"]
    # val_test = temp_dataset.train_test_split(test_size=0.5, seed=42)
    # val_dataset = val_test["train"]
    # test_dataset = val_test["test"]

    ###

    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    embedder = SentenceTransformer(embedding_model_name)

    predictions = []
    references = []
    prompts = []

    print("Evaluation Started ...")
    # 3) Generate content from prompts
    for example in test_dataset:
        # Updated by Nafis
        prompt_text = example["prompt"]
        reference_text = example["completion"]

        prompt = build_prompt(prompt_text)
        prediction = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
        )

        prompts.append(prompt)
        predictions.append(prediction)
        references.append(reference_text)

    # BLEU expects tokenized references as list-of-lists
    bleu_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        #print("Reference ...", references)
        if not pred or not ref:
            print("Skipping ... ", pred, ref)
            continue
        bleu_result = bleu_metric.compute(
            predictions=[pred],
            references=[[ref]],
        )
        rouge_result = rouge_metric.compute(
            predictions=[pred],
            references=[ref],
            use_aggregator=False,
        )

        bleu_scores.append(bleu_result["bleu"])
        rougeL_scores.append(rouge_result["rougeL"][0])

    cosine_scores = compute_cosine_similarity(
        embedder=embedder,
        predictions=predictions,
        references=references,
        device=embed_device,
    )

    avg_metrics = {
        "avg_bleu": sum(bleu_scores) / max(len(bleu_scores), 1),
        "avg_rougeL": sum(rougeL_scores) / max(len(rougeL_scores), 1),
        "avg_cosine_sim": sum(cosine_scores) / max(len(cosine_scores), 1),
    }

    print("\n=== Average Metrics ===")
    print(f"Average BLEU      : {avg_metrics['avg_bleu']:.4f}")
    print(f"Average ROUGE-L   : {avg_metrics['avg_rougeL']:.4f}")
    print(f"Average CosineSim : {avg_metrics['avg_cosine_sim']:.4f}")

    if save_predictions_path is not None:
        rows = []
        for prompt, pred, ref, bleu, rougeL, cos in zip(
            prompts, predictions, references, bleu_scores, rougeL_scores, cosine_scores
        ):
            rows.append({
                "prompt": prompt,
                "prediction": pred,
                "reference": ref,
                "bleu": bleu,
                "rougeL": rougeL,
                "cosine_similarity": cos,
            })

        os.makedirs(os.path.dirname(save_predictions_path), exist_ok=True)
        with open(save_predictions_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "average_metrics": avg_metrics,
                    "samples": rows,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    return avg_metrics


if __name__ == "__main__":
    MODEL_DIR = "saved_models/TestLLaMa-v1.0"
    TEST_DATASET_PATH = "Datasets/distill_data/train"

    evaluate_model(
        model_dir=MODEL_DIR,
        test_dataset_path=TEST_DATASET_PATH,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_new_tokens=256,
        save_predictions_path="saved_models/eval_predictions.json",
    )

    # saved_models/final_model_sft

    #saved_models/TestLLaMa-v1.0