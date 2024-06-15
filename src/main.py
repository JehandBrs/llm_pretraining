import argparse
import sys
from datasets import load_dataset
import torch

sys.path.append('drive/MyDrive/GPT2/src/')
from GPT2 import GPT2LMHeadModel, GPT2Config
from tokenizer import SimpleTokenizer
from Dataset import TextDataset
from training import train


parser = argparse.ArgumentParser(description='Train GPT2')

parser.add_argument('--datasets', type=list, default=['wikitext', 'wikitext-2-raw-v1'], help='The datasets to install from hugging face datasets')

parser.add_argument('--n_positions', type=int, default=1024, help='')

parser.add_argument('--n_ctx', type=int, default=1024, help='')

parser.add_argument('--n_embd', type=int, default=768, help='')

parser.add_argument('--n_layer', type=int, default=12, help='')

parser.add_argument('--n_head', type=int, default=12, help='')

parser.add_argument('--num_epochs', type=int, default=1, help='')

parser.add_argument('--batch_size', type=int, default=16, help='')

parser.add_argument('--device', type=str, default="cuda", help='')


def main(args):

  # Load dataset from hugging face
  dataset = load_dataset(args.datasets[0], args.datasets[1])

  # Build tokens vocab
  tokenizer = SimpleTokenizer()
  tokenizer.build_vocab(texts=[text['text'] for text in dataset['train']])
  vocab_size = len(tokenizer.vocab)
  print(f"The vocabulary created have {vocab_size} tokens")

  # Initialize GPT2 config
  config = GPT2Config(
      vocab_size=vocab_size, 
      n_positions=args.n_positions, 
      n_ctx=args.n_ctx, 
      n_embd=args.n_embd, 
      n_layer=args.n_layer, 
      n_head=args.n_head,
      )

  # Initialize the GPT2 model
  model = GPT2LMHeadModel(config)

  model_number_of_parameters = sum([param.numel() for param in model.parameters()])
  print(f"This model has {model_number_of_parameters} parameters")

  # Now we train this model
  num_epochs=args.num_epochs
  if "cuda" in args.device:
    device = args.device if torch.cuda.is_available() else "cpu"
  print(f"The device used is {device}")

  train(
    tokenizer=tokenizer, 
    model=model,
    dataset=dataset, 
    device=device,
    num_epochs=args.num_epochs,
    batch_size=args.batch_size
    )

  return

if __name__=="__main__":

  args = parser.parse_args()
  main(args)