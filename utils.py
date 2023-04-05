import torch
from transformers import GenerationConfig, LogitsWarper
from threading import Thread
from queue import Queue
import json

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

class CallbackLogitsWarper(LogitsWarper):
    def __init__(self, tokenizer, callback):
        self.tokenizer = tokenizer
        self.callback = callback
        self.res_tokens = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        self.res_tokens.append(input_ids[0][-1])
        result = self.tokenizer.decode(self.res_tokens).lstrip()
        result = result.split(':\n')

        if len(result) > 1:
          result = ':'.join(result[1:])
          if result != '':
            self.callback(result)

        return scores


def generate_streaming_completion(options):
  model = options.pop("model")
  tokenizer = options.pop("tokenizer")
  model_options = options.pop("model_options")
  stream = model_options.stream and model_options.num_beams == 1

  q = Queue() # fmin produces, the generator consumes


  generation_config = GenerationConfig(
    temperature=model_options.temperature,
    top_p=model_options.top_p,
    top_k=model_options.top_k,
    num_beams=model_options.num_beams,
    max_new_tokens=model_options.max_new_tokens,
  )

  prompt = generate_prompt(model_options.instruction, model_options.input)
  inputs = tokenizer(prompt, return_tensors="pt")
  input_ids = inputs["input_ids"].cuda()


  def stream_callback(res):
    q.put(json.dumps({ "text": res }) + '\n')

  logits_processor= [CallbackLogitsWarper(tokenizer, stream_callback)] if stream else None

  def generate():
    with torch.no_grad():
      model.eval()
      model.generate(
          input_ids=input_ids,
          generation_config=generation_config,
          logits_processor=logits_processor,
          return_dict_in_generate=True,
          # output_scores=True,
          # max_new_tokens=600
      )
      print('STREAMING DONE')
    torch.cuda.empty_cache()
    q.put("[DONE]")

  # start_new_thread(generate, ())
  Thread(target=generate,args=()).start()

  while True:
    next_item = q.get(True,10000) # Blocks until an input is available
    if next_item == "[DONE]":
        yield next_item
        break
    yield next_item


def generate_completion(options):
  model = options.pop("model")
  tokenizer = options.pop("tokenizer")
  model_options = options.pop("model_options")

  generation_config = GenerationConfig(
    temperature=model_options.temperature,
    top_p=model_options.top_p,
    top_k=model_options.top_k,
    num_beams=model_options.num_beams,
    max_new_tokens=model_options.max_new_tokens,
  )

  prompt = generate_prompt(model_options.instruction, model_options.input)
  inputs = tokenizer(prompt, return_tensors="pt")
  input_ids = inputs["input_ids"].cuda()

  with torch.no_grad():
    model.eval()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True
    )
    torch.cuda.empty_cache()

  output = tokenizer.decode(generation_output.sequences[0])
  del generation_output
  
  return {
    "text": output.split("### Response:")[1].strip()
  }