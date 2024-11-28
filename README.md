## A simple code to pre-train GPT-2 from scratch

Personnal project where I try to reimplement GPT-2 pretraining from scratch, by coding each element of the model and training procedure in PyTorch.

Here is how you can make it work (You can find an example on the notebook to run it on Colab) : 
```Shell
main.py \
--n_positions 1024 \
--n_ctx 1024 \
--n_embd 6 \
--n_layer 2 \
--n_head 2 \
--num_epochs 1 \
--batch_size 8
```
