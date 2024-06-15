import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
import torch.nn.functional as F
from Dataset import TextDataset
from tqdm import tqdm

def train(
  tokenizer, 
  model,
  dataset, 
  device='cpu',
  num_epochs=1,
  batch_size=1
  ):

  # Tokenizer
  if tokenizer==None:
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

  # Prepare the datasets
  train_texts = dataset['train']['text']
  test_texts = dataset['test']['text']

  train_dataset = TextDataset(train_texts, tokenizer)
  test_dataset = TextDataset(test_texts, tokenizer)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size)

  model.to(device)
  #model.half()
  model = torch.compile(model)
  # Optimizer and scheduler
  optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
  total_steps = len(train_loader) * num_epochs  # Assuming 1 epoch for simplicity
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_steps)

  # Training loop
  for epoch in range(num_epochs):
      model.train()
      total_loss = 0
      temp_loss = 0
      for i, batch in enumerate(tqdm(train_loader)):
          input_ids, attention_mask = batch
          input_ids = input_ids.to(device)
          attention_mask = attention_mask.to(device)
          
          outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
          loss = outputs[1]
          
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          scheduler.step()
          
          total_loss += loss.item()
          temp_loss += loss.item()
          if loss.item()==np.nan:
            print(nan)
          avg_train_loss = total_loss / len(train_loader)
          if i%250==0:
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {temp_loss:.2f}")
            temp_loss=0
      avg_train_loss = total_loss / len(train_loader)
      print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.2f}")
      return
      # Evaluation loop
      model.eval()
      total_eval_loss = 0
      with torch.no_grad():
          for batch in test_loader:
              input_ids, attention_mask = batch
              input_ids = input_ids.to(device)
              attention_mask = attention_mask.to(device)
              
              outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
              loss = outputs[1]
              
              total_eval_loss += loss.item()
              
      avg_eval_loss = total_eval_loss / len(test_loader)
      print(f"Epoch {epoch+1}/{num_epochs}, Evaluation Loss: {avg_eval_loss:.2f}")
      