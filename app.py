from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def calculate_if_math(text):
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

    formatted_prompt = f"Answer the question clearly: {prompt}"

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

while True:
    q = input("You: ")
    print("Hasa AI:", hasa_ai(q))