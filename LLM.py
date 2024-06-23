# This will contain the LLM 
from pathlib import Path
import torch

# Get the current script's directory
script_dir = Path(__file__).resolve().parent

# Define the relative path to the file from the script's directory
relative_path = Path('Data/shakespeare.txt')

# Construct the full path to the file
file_path = script_dir / relative_path

# Read the file
with file_path.open('r', encoding='utf-8') as file:
    text = file.read()

# Create a list with all characters in the text
chars = sorted(set(text))

# Create a dictionary to map characters to integers
string_to_int = { ch:i for i, ch in enumerate(chars) }
# Create a dictionary to map integers back to characters
int_to_string = { i:ch for i, ch in enumerate(chars) }

# Lambda function to encode a string to a list of integers
encode = lambda s: [string_to_int[c] for c in s]
# Lambda function to decode a list of integers back to a string
decode = lambda l: ''.join([int_to_string[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
print(data[:100])
