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
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import pandas as pd



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
from transformers import LlamaConfig, LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ================== Teacher-Student Evaluation Add-On ==================


# def evaluate_teacher_student(sentences, teacher_model, student_model, tokenizer, device, max_gen_len=50):
#     """
#     Generates outputs from teacher and student models for given sentences.
#     Computes BLEU, ROUGE-L, and cosine similarity of sentence embeddings.
#     Returns a list of dicts and prints outputs.
#     """
#     rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     embed_model = SentenceTransformer('all-MiniLM-L6-v2')
#     results = []

#     for i, sentence in enumerate(sentences):
#         inputs = tokenizer(sentence, return_tensors="pt").to(device)

#         # Teacher prediction
#         with torch.no_grad():
#             teacher_out = teacher_model.generate(**inputs, max_new_tokens=max_gen_len)
#         teacher_text = tokenizer.decode(teacher_out[0], skip_special_tokens=True)

#         # Student prediction
#         with torch.no_grad():
#             student_out = student_model.generate(**inputs, max_new_tokens=max_gen_len)
#         student_text = tokenizer.decode(student_out[0], skip_special_tokens=True)

#         # Compute metrics
#         bleu_score = sentence_bleu([teacher_text.split()], student_text.split())
#         rouge_score = rouge.score(teacher_text, student_text)['rougeL'].fmeasure
#         teacher_emb = embed_model.encode(teacher_text, convert_to_tensor=True)
#         student_emb = embed_model.encode(student_text, convert_to_tensor=True)
#         cosine_sim = util.cos_sim(teacher_emb, student_emb).item()

#         # Store results
#         results.append({
#             'sentence': sentence,
#             'teacher': teacher_text,
#             'student': student_text,
#             'bleu': bleu_score,
#             'rougeL': rouge_score,
#             'cosine_sim': cosine_sim
#         })

#         # Print for inspection
#         print(f"\nSample {i+1}:")
#         print("Input:", sentence)
#         print("Teacher (ground truth):", teacher_text)
#         print("Student (predicted)  :", student_text)
#         print(f"BLEU: {bleu_score:.4f}, ROUGE-L: {rouge_score:.4f}, CosineSim: {cosine_sim:.4f}")

#     return results

def evaluate_teacher_student(
    sentences,
    teacher_model,
    student_model,
    tokenizer,
    device,
    max_gen_len=50
):
    """
    Generates outputs from teacher and student models for given sentences.
    Computes BLEU, ROUGE-L, and cosine similarity of sentence embeddings.
    Returns:
        - per-sample results
        - averaged metrics across all samples
    """
    import torch
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer, util

    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    smooth = SmoothingFunction().method1

    results = []

    total_bleu = 0.0
    total_rougeL = 0.0
    total_cosine_sim = 0.0

    for i, sentence in enumerate(sentences):
        inputs = tokenizer(sentence, return_tensors="pt").to(device)

        # Teacher prediction
        with torch.no_grad():
            teacher_out = teacher_model.generate(**inputs, max_new_tokens=max_gen_len)
        teacher_text = tokenizer.decode(teacher_out[0], skip_special_tokens=True)

        # Student prediction
        with torch.no_grad():
            student_out = student_model.generate(**inputs, max_new_tokens=max_gen_len)
        student_text = tokenizer.decode(student_out[0], skip_special_tokens=True)

        # Compute metrics
        bleu_score = sentence_bleu(
            [teacher_text.split()],
            student_text.split(),
            smoothing_function=smooth
        )
        rouge_score = rouge.score(teacher_text, student_text)['rougeL'].fmeasure
        teacher_emb = embed_model.encode(teacher_text, convert_to_tensor=True)
        student_emb = embed_model.encode(student_text, convert_to_tensor=True)
        cosine_sim = util.cos_sim(teacher_emb, student_emb).item()

        total_bleu += bleu_score
        total_rougeL += rouge_score
        total_cosine_sim += cosine_sim

        # Store results
        results.append({
            'sentence': sentence,
            'teacher': teacher_text,
            'student': student_text,
            'bleu': bleu_score,
            'rougeL': rouge_score,
            'cosine_sim': cosine_sim
        })

        # Print for inspection
        print(f"\nSample {i+1}:")
        print("Input:", sentence)
        print("Teacher (ground truth):", teacher_text)
        print("Student (predicted)  :", student_text)
        print(f"BLEU: {bleu_score:.4f}, ROUGE-L: {rouge_score:.4f}, CosineSim: {cosine_sim:.4f}")

    num_samples = len(results)
    avg_metrics = {
        'avg_bleu': total_bleu / num_samples if num_samples > 0 else 0.0,
        'avg_rougeL': total_rougeL / num_samples if num_samples > 0 else 0.0,
        'avg_cosine_sim': total_cosine_sim / num_samples if num_samples > 0 else 0.0,
    }

    print("\n=== Average Metrics ===")
    print(f"Average BLEU      : {avg_metrics['avg_bleu']:.4f}")
    print(f"Average ROUGE-L   : {avg_metrics['avg_rougeL']:.4f}")
    print(f"Average CosineSim : {avg_metrics['avg_cosine_sim']:.4f}")

    return results, avg_metrics

def evaluate_and_save(student_model, teacher_model, tokenizer, device, result_dir, num_samples=10):
    """
    Helper to automatically run evaluation on WikiText samples for a student model
    and save results to CSV.
    """
    # Take first N non-empty sentences from WikiText train split
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    sentences = [s for s in dataset["text"] if len(s.strip()) > 20][:num_samples]

    print(f"\n=== Evaluating Student Model")
    results = evaluate_teacher_student(sentences, teacher_model, student_model, tokenizer, device)

    # Save results to CSV
    csv_file = os.path.join(result_dir, f"student_layer_eval.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    print(f"Evaluation CSV saved to {csv_file}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("Starting inference on GPU" if torch.cuda.is_available() else "Starting inference on CPU")
result_dir = "eval_results"


student_model_name = "saved_models/student_model_student_layers"
teacher_model_name = "meta-llama/Llama-3.2-3B"

student_model = AutoModelForCausalLM.from_pretrained(student_model_name).to(device)
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
teacher_model.eval()


evaluate_and_save(student_model, teacher_model, tokenizer, device, result_dir, num_samples=1000)