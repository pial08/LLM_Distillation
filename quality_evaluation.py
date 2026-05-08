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
from datasets import load_dataset, concatenate_datasets
import pandas as pd

#  test on squad dataset, and wikitext dataset

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
import evaluate
from openai import OpenAI
from google import genai
import re
from dotenv import load_dotenv




OPENAI_JUDGE_MODEL = "gpt-5-nano"
GEMINI_JUDGE_MODEL = "gemini-3.1-pro-preview"
load_dotenv() 


def load_config(config_path):
    """Loads configuration parameters from a JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def extract_json(text):
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Could not parse JSON from model response:\n{text}")



def prepare_mixed_dataset(
    tokenizer,
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
    combined = concatenate_datasets([wikitext, squad]).shuffle(seed=seed)
    return combined

    # 6) Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized_dataset = combined.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return tokenized_dataset



def build_judge_prompt(input_sentence, teacher_text, student_text):
    return f"""
        You are an expert evaluator for teacher-student knowledge distillation.

        Compare the student output against the teacher output for the same input. Use the below guidelines to judge the teacher output vs the student output.
        Be gentle with scoring, 

        Input sentence:
        {input_sentence}

        Teacher output:
        {teacher_text}

        Student output:
        {student_text}

        Return JSON only with these fields:
        {{
        "overall_score": number from 0 to 5
        }}

        Scoring rules:
        5 = nearly identical or fully equivalent in meaning or token matching
        4 = mostly equivalent with minor differences
        3 = partially similar but missing/changing some meaning
        2 = weakly related
        1 = mostly unrelated
        0 = completely unrelated or empty
    """

def gpt_judge(input_sentence, teacher_text, student_text):
    #api_key = os.environ.get("OPENAI_API_KEY")
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)

    prompt = build_judge_prompt(input_sentence, teacher_text, student_text)

    response = client.responses.create(
        model=OPENAI_JUDGE_MODEL,
        input=prompt
    )

    return extract_json(response.output_text)



def gemini_judge(input_sentence, teacher_text, student_text):
    #api_key = os.environ.get("GEMINI_API_KEY")
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return None

    client = genai.Client(api_key=api_key)

    prompt = build_judge_prompt(input_sentence, teacher_text, student_text)

    response = client.models.generate_content(
        model=GEMINI_JUDGE_MODEL,
        contents=prompt
    )

    return extract_json(response.text)


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
    bertscore_metric = evaluate.load("bertscore")
    smooth = SmoothingFunction().method1

    results = []

    # total_bleu = 0.0
    # total_rougeL = 0.0
    # total_cosine_sim = 0.0

    # token matching metrics
    total_bleu = 0.0
    total_rougeL = 0.0
    
    # embedding matching metrics
    total_cosine_sim = 0.0
    total_bertscore_f1 = 0.0
    
    # llm evaluation
    total_gpt_score = 0.0
    total_gemini_score = 0.0

    gpt_count = 0
    gemini_count = 0

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

        bertscore_result = bertscore_metric.compute(
            predictions=[student_text],
            references=[teacher_text],
            lang="en"
        )


        gpt_result = None
        gemini_result = None

        gpt_score = None
        gemini_score = None

        try:
            gpt_result = gpt_judge(sentence, teacher_text, student_text)
            if gpt_result is not None:
                gpt_score = float(gpt_result["overall_score"])
                total_gpt_score += gpt_score
                gpt_count += 1
        except Exception as e:
            gpt_result = {"error": str(e)}

        try:
            gemini_result = gemini_judge(sentence, teacher_text, student_text)
            if gemini_result is not None:
                gemini_score = float(gemini_result["overall_score"])
                total_gemini_score += gemini_score
                gemini_count += 1
        except Exception as e:
            gemini_result = {"error": str(e)}

        


        rouge_score = rouge.score(teacher_text, student_text)['rougeL'].fmeasure
        teacher_emb = embed_model.encode(teacher_text, convert_to_tensor=True)
        student_emb = embed_model.encode(student_text, convert_to_tensor=True)
        cosine_sim = util.cos_sim(teacher_emb, student_emb).item()
        bert_score_f1 = bertscore_result["f1"][0]


        total_bleu += bleu_score
        total_rougeL += rouge_score
        total_cosine_sim += cosine_sim
        total_bertscore_f1 += bert_score_f1

        # Store results
        results.append({
            'sentence': sentence,
            'teacher': teacher_text,
            'student': student_text,
            'bleu': bleu_score,
            'rougeL': rouge_score,
            'cosine_sim': cosine_sim,
            'bertscore': bert_score_f1,
            "gpt_overall_score": gpt_score
            #"gemini_overall_score": gemini_score
        })

        # Print for inspection
        print(f"\nSample {i+1}:")
        print("Input:", sentence)
        print("Teacher (ground truth):", teacher_text)
        print("Student (predicted)  :", student_text)
        #print(f"BLEU: {bleu_score:.4f}, ROUGE-L: {rouge_score:.4f}, CosineSim: {cosine_sim:.4f}, BERTScore_F1: {bert_score_f1:.4f}, GPT Score: {gpt_score:.4f}, GEMINI Score: {gemini_score:.4f}")
        print(f"BLEU: {bleu_score:.4f}, ROUGE-L: {rouge_score:.4f}, CosineSim: {cosine_sim:.4f}, BERTScore_F1: {bert_score_f1:.4f}")

        if gpt_score is not None:
            print(f"GPT Judge Score: {gpt_score:.2f}/5")
        else:
            print("GPT Judge Score: skipped or failed")

        if gemini_score is not None:
            print(f"Gemini Judge Score: {gemini_score:.2f}/5")
        else:
            print("Gemini Judge Score: skipped or failed")

    num_samples = len(results)
    avg_metrics = {
        'avg_bleu': total_bleu / num_samples if num_samples > 0 else 0.0,
        'avg_rougeL': total_rougeL / num_samples if num_samples > 0 else 0.0,
        'avg_cosine_sim': total_cosine_sim / num_samples if num_samples > 0 else 0.0,
        'avg_bert_score_f1': total_bertscore_f1 / num_samples if num_samples else 0.0,
        'avg_gpt_judge_score': total_gpt_score / gpt_count if gpt_count else None,
        'avg_gemini_judge_score': total_gemini_score / gemini_count if gemini_count else None,
        'gpt_judged_samples': gpt_count,
        'gemini_judged_samples': gemini_count
    }



    

    print("\n=== Average Metrics ===")
    print(f"Average BLEU      : {avg_metrics['avg_bleu']:.4f}")
    print(f"Average ROUGE-L   : {avg_metrics['avg_rougeL']:.4f}")
    print(f"Average CosineSim : {avg_metrics['avg_cosine_sim']:.4f}")
    print(f"Average BERTScore : {avg_metrics['avg_bert_score_f1']:.4f}")
    print(f"Average GPT Score : {avg_metrics['avg_gpt_judge_score']:.4f}")
    print(f"Average GEMINI Score : {avg_metrics['avg_gemini_judge_score']:.4f}")

    return results, avg_metrics

def evaluate_and_save(student_model, teacher_model, tokenizer, device, result_dir, num_samples=10):
    """
    Helper to automatically run evaluation on WikiText samples for a student model
    and save results to CSV.
    """
    # Take first N non-empty sentences from WikiText train split

    dataset = prepare_mixed_dataset(
            tokenizer=tokenizer,
            max_length=256,
            total_samples=100,
            wikitext_ratio=0.6,
            seed=42,
            split="validation",
        )
    #dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    sentences = [s for s in dataset["text"] if len(s.strip()) > 20][:num_samples]

    print(f"\n=== Evaluating Student Model")
    results = evaluate_teacher_student(sentences, teacher_model, student_model, tokenizer, device)

    # Save results to CSV
    csv_file = os.path.join(result_dir, f"student_layer_eval.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    print(f"Evaluation CSV saved to {csv_file}")



config_path = "config.json"
config = load_config(config_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("Starting inference on GPU" if torch.cuda.is_available() else "Starting inference on CPU")
result_dir = "eval_results"



student_model_name = "saved_models/TestLLaMa-v1.0"
#teacher_model_name = "meta-llama/Llama-3.1-8B-Instruct"

teacher_model_name = config.get("teacher_model_name")
print("Printing Teacher and Student Model name", teacher_model_name, student_model_name)

# ./gkd_out/RL_Distil_1.2
# saved_models/student_model_student_layers
# saved_models/TestLLaMa-v1.0


student_model = AutoModelForCausalLM.from_pretrained(student_model_name).to(device)



#original teacher model leading
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
teacher_model.eval()


# # Load teacher using funciton
# # Load configuration
# config_path = "config.json"
# config = load_config(config_path)

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # logging.info("Starting training on GPU" if torch.cuda.is_available() else "Starting training on CPU")

# # Load teacher model and tokenizer (original) -- commenting temp
# teacher_model_name = config.get("teacher_model_name", "distilgpt2")
# teacher_model, tokenizer = create_teacher_model(config)


evaluate_and_save(student_model, teacher_model, tokenizer, device, result_dir, num_samples=50)