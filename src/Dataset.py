from torch.utils.data import Dataset, DataLoader

# Custom dataset to encode input text
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings_dict = self.tokenizer('<|startoftext|>' + text + '<|endoftext|>', 
                                       truncation=True,
                                       max_length=self.max_length, 
                                       padding="max_length", 
                                       return_tensors='pt')
        input_ids = encodings_dict['input_ids'].squeeze()
        attention_mask = encodings_dict['attention_mask'].squeeze()
        return input_ids, attention_mask