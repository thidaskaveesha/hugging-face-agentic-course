# we need to turn messages into prompts for the model
# we can do this using tokenizer and some formatting rules
from transformers import AutoTokenizer

# model cannot directly consume this
messages = [
    {"role": "system", "content": "You are an AI assistant with access to various tools."},
    {"role": "user", "content": "Hi !"},
    {"role": "assistant", "content": "Hi human, what can help you with ?"},
]
# convert chat messages into a single text prompt that the model was trained to understand 
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# output will be like below 
# models have metadata depending on model it changes the formatting rules
# thats why we initialize tokenizer from the model
# Output:
#    <|system|> You are an AI assistant. <|user|> Hi! <|assistant|> Hi human, what can I help you with? <|assistant|> 

