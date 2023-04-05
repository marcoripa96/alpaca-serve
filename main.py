from fastapi import FastAPI, Body, Response
from fastapi.responses import StreamingResponse
from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig
from utils import generate_streaming_completion, generate_completion
from pydantic import BaseModel, Field


tokenizer = LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LLaMAForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "tloen/alpaca-lora-7b", device_map={'': 0})

class CompletionRequest(BaseModel):
  instruction: str = Field(description="Describes the task the model should perform.", required=True)
  input: str = Field(default=None, description="Optional context or input for the task.", required=False)
  temperature: float = Field(default=0.7, ge=0, le=2, description="The value used to modulate the next token probabilities.", required=False)
  top_k: int = Field(default=50, description="The number of highest probability vocabulary tokens to keep for top-k-filtering.", required=False)
  top_p: float = Field(default=0.9, description="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.", required=False)
  num_beams: int = Field(default=1, description="Number of beams for beam search. 1 means no beam search.", required=False)
  max_new_tokens: int = Field(default=250, description="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.", required=False)
  stream: bool = Field(default=False, description="The response is an Event stream. If num_beans > 1 this option is set to False.", required=False)

app = FastAPI()

@app.post("/completion")
async def completion(res: Response, model_options: CompletionRequest = Body()):

  generate_options = {
    "model": model,
    "tokenizer": tokenizer,
    "model_options": model_options
  }

  stream = model_options.stream and model_options.num_beams == 1

  if stream:
    return StreamingResponse(generate_streaming_completion(generate_options))

  return generate_completion(generate_options)


