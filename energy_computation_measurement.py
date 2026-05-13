import os
import time
import threading
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional whole-run energy estimate
USE_CODECARBON = False
if USE_CODECARBON:
    from codecarbon import EmissionsTracker

# Optional direct NVIDIA GPU energy measurement
USE_NVML = True
if USE_NVML:
    import pynvml

def load_config(config_path):
    """Loads configuration parameters from a JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

@dataclass
class PromptMetrics:
    prompt: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_sec: float
    tokens_per_sec: float
    peak_gpu_mem_gb: float
    gpu_energy_joules: float | None
    generated_text: str


class NVMLPowerMonitor:
    """
    Polls GPU power usage using NVML and integrates power over time:
    Energy (J) = integral of Power(W) over time(s).
    """
    def __init__(self, gpu_index: int = 0, interval_sec: float = 0.05):
        self.gpu_index = gpu_index
        self.interval_sec = interval_sec
        self._running = False
        self._thread = None
        self.energy_joules = 0.0
        self.samples = []

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    def _loop(self):
        last_t = time.perf_counter()
        while self._running:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)  # milliwatts
            now = time.perf_counter()
            dt = now - last_t
            power_w = power_mw / 1000.0
            self.energy_joules += power_w * dt
            self.samples.append((now, power_w))
            last_t = now
            time.sleep(self.interval_sec)

    def start(self):
        self.energy_joules = 0.0
        self.samples = []
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()

    def shutdown(self):
        pynvml.nvmlShutdown()


def load_local_causal_lm(
    model_dir: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        device_map=device_map,
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer


def run_inference_with_metrics(
    model_dir: str,
    prompts: List[str],
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    gpu_index: int = 0,
) -> List[PromptMetrics]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_local_causal_lm(model_dir)

    # Optional experiment-level tracker
    tracker = None
    if USE_CODECARBON:
        tracker = EmissionsTracker(save_to_file=False, log_level="error")
        tracker.start()

    power_monitor = None
    if USE_NVML and torch.cuda.is_available():
        power_monitor = NVMLPowerMonitor(gpu_index=gpu_index)

    results: List[PromptMetrics] = []

    for prompt in prompts:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Move tokenized inputs to the model's primary device
        model_device = next(model.parameters()).device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)

        input_tokens = input_ids.shape[1]

        if power_monitor is not None:
            power_monitor.start()

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        end = time.perf_counter()

        if power_monitor is not None:
            power_monitor.stop()
            gpu_energy_joules = power_monitor.energy_joules
        else:
            gpu_energy_joules = None

        generated_ids = outputs[0]
        total_tokens = generated_ids.shape[0]
        output_tokens = total_tokens - input_tokens

        latency_sec = end - start
        tokens_per_sec = output_tokens / latency_sec if latency_sec > 0 else 0.0

        if torch.cuda.is_available():
            peak_gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            peak_gpu_mem_gb = 0.0

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        results.append(
            PromptMetrics(
                prompt=prompt,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_sec=latency_sec,
                tokens_per_sec=tokens_per_sec,
                peak_gpu_mem_gb=peak_gpu_mem_gb,
                gpu_energy_joules=gpu_energy_joules,
                generated_text=generated_text,
            )
        )

    if tracker is not None:
        emissions_kg = tracker.stop()
        print(f"\nEstimated total experiment emissions (kg CO2eq): {emissions_kg:.6f}")

    if power_monitor is not None:
        power_monitor.shutdown()

    return results


def summarize_results(results: List[PromptMetrics]) -> Dict[str, Any]:
    n = len(results)
    avg_latency = sum(r.latency_sec for r in results) / n
    avg_tps = sum(r.tokens_per_sec for r in results) / n
    avg_peak_mem = sum(r.peak_gpu_mem_gb for r in results) / n

    valid_energy = [r.gpu_energy_joules for r in results if r.gpu_energy_joules is not None]
    avg_energy = sum(valid_energy) / len(valid_energy) if valid_energy else None

    total_output_tokens = sum(r.output_tokens for r in results)
    total_latency = sum(r.latency_sec for r in results)
    corpus_tps = total_output_tokens / total_latency if total_latency > 0 else 0.0

    energy_per_token = (
        sum(valid_energy) / total_output_tokens
        if valid_energy and total_output_tokens > 0
        else None
    )

    return {
        "num_prompts": n,
        "avg_latency_sec": avg_latency,
        "avg_tokens_per_sec": avg_tps,
        "corpus_tokens_per_sec": corpus_tps,
        "avg_peak_gpu_mem_gb": avg_peak_mem,
        "avg_gpu_energy_joules": avg_energy,
        "gpu_energy_per_token_joules": energy_per_token,
    }


if __name__ == "__main__":

    config_path = "config.json"
    config = load_config(config_path)

    "meta-llama/Llama-Guard-3-1B"
    teacher_model_name = config.get("teacher_model_name")
    #saved_model_name = "saved_models/TestLLaMa-v1.0"

    saved_model_name = "saved_models/TestLLaMa-v1.0"
    #model_dir = "mistralai/Mistral-7B-Instruct-v0.3"


    

    prompts = [
        "This ammunition , and that which I brought with me , was rapidly prepared for use at the Laboratory", 
        "Partly due to these events , and partly due to the major losses in manpower Gallia suffers towards the end of the war with the Empire , the Nameless are offered a formal position as a squad in the Gallian Army rather than serve as an anonymous shadow force .",
        "In a preview of the TGS demo , Ryan Geddes of IGN was left excited as to where the game would go after completing the demo , along with enjoying the improved visuals over Valkyria Chronicles II",
        "Summarize the concept of knowledge distillation in LLMs.",
        "Write a short paragraph about why latency matters in model deployment.",
        "For several years the arsenal , which was owned by the federal government , served as a simple arms depot and was staffed with only a handful of soldiers . But in November 1860 , with the American Civil War on the horizon",
    ]


    
    print("Outupt for the base model ........", teacher_model_name)
    results = run_inference_with_metrics(
        model_dir=teacher_model_name,
        prompts=prompts,
        max_new_tokens=80,
        do_sample=False,   # use deterministic decoding for fair comparisons
        temperature=0.7,
        top_p=0.9,
        gpu_index=0,
    )

    print("\nPer-prompt results:")
    for r in results:
        print(asdict(r))
        print("-" * 80)

    summary = summarize_results(results)
    print("\nSummary:")
    for k, v in summary.items():
        print(f"{k}: {v}")



    print("Outupt for the saved models ........", saved_model_name)
    results = run_inference_with_metrics(
        model_dir=saved_model_name,
        prompts=prompts,
        max_new_tokens=80,
        do_sample=False,   # use deterministic decoding for fair comparisons
        temperature=0.7,
        top_p=0.9,
        gpu_index=0,
    )

    print("\nPer-prompt results:")
    for r in results:
        print(asdict(r))
        print("-" * 80)

    summary = summarize_results(results)
    print("\nSummary:")
    for k, v in summary.items():
        print(f"{k}: {v}")
    
