from smolagents import LiteLLMModel 

# create local hosted ollama based model 
model = LiteLLMModel(
    model_id="ollama_chat/qwen2:7b",
    base_url="http://localhost:11434",
    num_ctx=8192,
)

def generate_response(prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]
    response = model.generate(messages)
    # Extract plain text from the first choice
    return response.content if isinstance(response.content, str) else response.content[0]["text"]

# Example usage
if __name__ == "__main__":
    prompt = "Hello, how are you?"
    print(generate_response(prompt))