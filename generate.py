import argparse
import requests
import json

def stream_completion(options: dict):
  import curses
  # Initialize the curses module
  stdscr = curses.initscr()
  curses.echo()
  curses.cbreak()

  instruction_opt = options['instruction']

  def generate(instruction = None):
    stdscr.clear()
    if not instruction:
      stdscr.addstr(0, 0, "Prompt: ")
      # Get user input
      instruction = stdscr.getstr(1, 0)
    res = requests.post('http://127.0.0.1:8000/completion', json={**options, "stream": True}, stream=True)

    # Clear the screen before printing new output
    stdscr.clear()
    for chunk in res.iter_lines():
        if chunk:
          decoded_chunk = chunk.decode("utf-8")
          if decoded_chunk == '[DONE]':
            break
          parsed_chunk = json.loads(decoded_chunk)
          # Move the cursor to the beginning of the line and overwrite the previous output
          stdscr.addstr(0, 0, parsed_chunk['text'])
          stdscr.refresh()

  try:
    if instruction_opt:
      generate(instruction=instruction_opt)
      stdscr.getch() 
    else:
      while True:
        generate(instruction=instruction_opt)
        print('\n')
        
  finally:
      # Clean up the curses module
      curses.endwin()

def completion(options: dict):
  instruction_opt = options['instruction']
  
  def generate(instruction=None):
    if not instruction:
      instruction = input("Prompt: ")
    res = requests.post('http://127.0.0.1:8000/completion', json={**options, "stream": False})

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





