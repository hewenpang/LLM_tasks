from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# FastAPI app initialization
app = FastAPI()

# Load model and tokenizer
model_name = "Qwen/Qwen3-4B-Thinking-2507"
save_directory = "D:\\large_models"

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=save_directory)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=save_directory)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class ChatRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_text(request: ChatRequest):
    prompt = request.prompt
    messages = [{"role": "user", "content": prompt}]
    
    # Tokenize the input text
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate text
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2000
    )
    
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    
    try:
        # Find the position of 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return {"thinking_content": thinking_content, "content": content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
