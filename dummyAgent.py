import os
from huggingface_hub import InferenceClient

## You need a token from https://hf.co/settings/tokens, ensure that you select 'read' as the token type. If you run this on Google Colab, you can set it up in the "settings" tab under "secrets". Make sure to call it "HF_TOKEN"
# os.environ.set("HF_TOKEN", "your_huggingface_token_here")  # replace with your actual token or set it in your environment
HF_TOKEN = os.environ.get("HF_TOKEN")

client = InferenceClient(model="meta-llama/Llama-4-Scout-17B-16E-Instruct")


output = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "The capital of France is"},
    ],
    stream=False,
    max_tokens=20,
)
print(output.choices[0].message.content)

# This system prompt is a bit more complex and actually contains the function description already appended.
# Here we suppose that the textual description of the tools have already been appended
SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
get_weather: Get the current weather in a given location, args: {{"location": {{"type": "string"}}}}
example use :
```
{{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}}

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:
```
$JSON_BLOB
```
Observation: the result of the action. This Observation is unique, complete, and the source of truth.
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer. """

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in London?"},
]

output = client.chat.completions.create(
    messages=messages,
    stream=False,
    max_tokens=200,
)
print(output.choices[0].message.content)

# The answer was hallucinated by the model. We need to stop to actually execute the function!
output = client.chat.completions.create(
    messages=messages,
    max_tokens=150,
    stop=["Observation:"] # Let's stop before any actual function is called
)

print(output.choices[0].message.content)

# Dummy function
def get_weather(location):
    return f"the weather in {location} is sunny with low temperatures. \n"

get_weather('London')

# Let's concatenate the base prompt, the completion until function execution and the result of the function as an Observation
messages=[
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in London ?"},
    {"role": "assistant", "content": output.choices[0].message.content+"Observation:\n"+get_weather('London')},
]
messages

output = client.chat.completions.create(
    messages=messages,
    stream=False,
    max_tokens=200,
)

print(output.choices[0].message.content)