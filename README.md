# Sinusoidal Position Embeddings visualized

Sinusoidal Positional encodings are, famously, used in the Transformer models ([Vaswani, et al.](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf)) to encode the positions of tokens in a sequence.

There are parallels to be drawn from binary representation. Indeed, [this](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) excellent post by Amirhossein Kazemnejad explains.

This repository contains code to generate some simple visualizations of these encodings.

### For an information density of 500:
![](position_embeddings_500.gif)

### For an information density of 10000:
![](position_embeddings_10000.gif)

Notice how changing reducing the information density shifts 'weight' to larger indices more quickly.
