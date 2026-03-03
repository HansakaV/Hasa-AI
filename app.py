from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def calculate_if_math(text):
    # detect simple math expressions
    if re.fullmatch(r"[0-9\s\+\-\*/\.]+", text.strip()):
        try:
            return str(eval(text))
        except:
            return None
    return None

def hasa_ai(prompt):
    math_result = calculate_if_math(prompt)
    if math_result is not None:
        return math_result

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

while True:
    q = input("You: ")
    print("Hasa AI:", hasa_ai(q))