import argparse
import requests
import json
import sys

def stream_completion(options: dict):
  instruction_opt = options['instruction']

  def generate(instruction = None):
    if not instruction:
    # Get user input
      instruction = input('Prompt: ')
    
    resolved_options = {
      **options,
      "instruction": instruction
    }

    res = requests.post('http://127.0.0.1:7860/completion', json={**resolved_options, "stream": True}, stream=True)

    for chunk in res.iter_lines():
        if chunk:
          decoded_chunk = chunk.decode("utf-8")
          parsed_chunk = json.loads(decoded_chunk)
          sys.stdout.write(parsed_chunk['text'])
          sys.stdout.flush()
          
    sys.stdout.write('\n')
    sys.stdout.flush()

  # try:
  if instruction_opt:
    generate(instruction=instruction_opt)
  else:
    while True:
      generate(instruction=instruction_opt)
      print('\n')
        

def completion(options: dict):
  instruction_opt = options['instruction']
  
  def generate(instruction=None):
    if not instruction:
      instruction = input("Prompt: ")

    resolved_options = {
      **options,
      "instruction": instruction
    }
    res = requests.post('http://127.0.0.1:8000/completion', json={**resolved_options, "stream": False})

    print(res.json()['text'])

  if instruction_opt:
    generate(instruction=instruction_opt)
  else:
    while True:
      generate(instruction=instruction_opt)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--stream', action='store_true')
  parser.add_argument('--no-stream', dest='stream', action='store_false')
  parser.set_defaults(feature=False)
  parser.add_argument('--p', dest='prompt', default=None, type=str)
  args = parser.parse_args()

  prompt = args.prompt

  options = {
    "instruction": prompt
  }

  if args.stream:
    stream_completion(options)
  else:
    completion(options)





