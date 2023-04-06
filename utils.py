import torch
from transformers import GenerationConfig, LogitsWarper, TextIteratorStreamer
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


def tokenize(options):
  tokenizer = options.pop("tokenizer")
  tokenizer_options = options.pop("tokenizer_options")
  prompt = generate_prompt(tokenizer_options.text, None)
  inputs = tokenizer(prompt, return_tensors="pt")
  return inputs


def generate_streaming_completion(options):
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
  streamer = TextIteratorStreamer(tokenizer,skip_prompt=True)

  def generate():
    with torch.no_grad():
      model.eval()
      model.generate(
          input_ids=input_ids,
          generation_config=generation_config,
          streamer=streamer,
          return_dict_in_generate=True,
      )
      print('STREAMING DONE')
    torch.cuda.empty_cache()

  Thread(target=generate,args=()).start()

  for index, new_text in enumerate(streamer):
      if index > 0:
        if new_text:
          yield json.dumps({ "text": new_text }) + '\n'
        


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