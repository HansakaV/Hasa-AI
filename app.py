import wikipedia
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# ----------------------
# Math Tool
# ----------------------
def calculate_if_math(text):
    if re.fullmatch(r"[0-9\s\+\-\*/\.]+", text.strip()):
        try:
            return str(eval(text))
        except Exception:
            return None
    return None


# ----------------------
# Wikipedia Tool
# ----------------------
def search_wikipedia(query):
    try:
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except Exception:
        return None


# ----------------------
# Hasa AI Core
# ----------------------

conversation_memory = []


def hasa_ai(prompt):

    global conversation_memory

    math_result = calculate_if_math(prompt)
    if math_result:
        return math_result

    wiki_result = search_wikipedia(prompt)
    if wiki_result:
        return wiki_result

    conversation_memory.append(prompt)
    conversation_context = " ".join(conversation_memory[-5:])

    formatted_prompt = f"""
You are Hasa AI.
You are aggressive, confident, sharp.
Context: {conversation_context}
Answer clearly:
{prompt}
"""

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    conversation_memory.append(response)

    return response


# Chat Loop
while True:
    q = input("You: ")
    print("Hasa AI:", hasa_ai(q))
