import torch
import torch.nn as nn
import math

class GPT2Config:
    def __init__(self, vocab_size=50257, n_positions=1024, n_ctx=1024, n_embd=768, n_layer=12, n_head=12):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head

class LayerNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_embd))
        self.b = nn.Parameter(torch.zeros(n_embd))
        self.eps = eps

    def forward(self, x):
        return self.g * (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + self.eps) + self.b

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.split_size = config.n_embd
        self.scale = math.sqrt(config.n_embd // config.n_head)

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

    def forward(self, x, layer_past=None, attention_mask=None):
        query, key, value = self.c_attn(x).split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value)

        attn_output, attn_weights = self.scaled_dot_product_attention(query, key, value, attention_mask)
        attn_output = self.merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output, present

    def split_heads(self, x):
        new_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_shape)

    def scaled_dot_product_attention(self, query, key, value, attention_mask):
        scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale
        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(1).unsqueeze(2)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.act(self.c_fc(x))
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, layer_past=None, attention_mask=None):
        output_attn, present = self.attn(self.ln_1(x), layer_past, attention_mask)
        x = x + output_attn
        x = x + self.mlp(self.ln_2(x))
        return x, present

class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(0.1)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd)

    def forward(self, input_ids, position_ids=None, attention_mask=None, past=None):
        if past is None:
            past = [None] * len(self.h)

        if position_ids is None:
            position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_embeds = self.wte(input_ids) + self.wpe(position_ids)
        hidden_states = self.drop(input_embeds)

        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past, attention_mask)
            presents.append(present)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states, presents

class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids, position_ids=None, attention_mask=None, labels=None):
        hidden_states, presents = self.transformer(input_ids, position_ids, attention_mask)
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return lm_logits, loss
