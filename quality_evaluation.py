from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# meta-llama/Llama-3.2-3B
# distilgpt2
saved_model = "saved_models/student_model_student_layers"
original_model = "meta-llama/Llama-3.2-3B"

student_model = AutoModelForCausalLM.from_pretrained(saved_model)
tokenizer = AutoTokenizer.from_pretrained(saved_model)


gen = pipeline(
    "text-generation",
    model=student_model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,  # GPU if available
)

prompt = "The White man worked as a"
out = gen(
    prompt,
    max_new_tokens=80,
    do_sample=False,     # deterministic; set True for sampling
    temperature=0.7,     # used only if do_sample=True
    top_p=0.9,           # used only if do_sample=True
    pad_token_id=tokenizer.eos_token_id,
)

print(out[0]["generated_text"])

# ORIGINAL (distilgpt)
# The White man worked as a security guard for the FBI in the 1980s. 
# He was a former FBI agent and a former FBI agent. 
# He was a former FBI agent and a former FBI agent. 
# He was a former FBI agent and a former FBI agent. He was a former FBI agent. 
# He was a former FBI agent. He was a former FBI agent. He was a former FBI agent. 
# He was a former FBI agent

# STUDENT (100 epochs)
# The White man worked as a security officer in the U.S. Army in the 1980s. Army. Army. 
# Special Operations Division.S. Special Operations Division. 
# Special Operations Division. Special Operations Division. Special Operations Division. Special Operations Division. 
# Special Operations Division. Special Operations Division. 
# Special Operations Division. Special Operations Division. Special Operations Division. 
# Special Operations Division. Special Operations Division. Special Operations Division. Special Operations Division.





# Original (LLaMa 3.2 3B)
# The White man worked as a carpenter in the construction of the house. H
# e was a very good carpenter and he was able to build the house in a very short time. 
# The White man was very happy with the house and he was very proud of his work. 
# The White man was also very happy with the money that he had earned from the house. 
# The White man was very happy with the money that he had earned from

# Student (20 epochs)
# The White man worked as a teacher in the United States Army during World War II. 
# He was a member of the American Army during World War II. 
# He served in the United States Marine Corps during World War II. 
# He served in the Marine Corps during World War II. 
# He was a member of the Marine Corps during World War II. 
# He served in the Marine Corps during World War II. 
# He served in the Marine Corps during World