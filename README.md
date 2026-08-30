Hi, this package is in development, not ready yet.

It is being tested and most stuff already works.

Needs more layers and archictectures.

Here is how it works:


```mermaid
flowchart LR
    A[You] -->|Want convenience| B(Pytorch)
    B -->|Works slowly because adds python overhead| C[CUDA]

    D[Me] --> |Want to do math| E(Rewrite popular hf models in JAX)
    E --> |Models convert to pytorch through a simple parser and prebuilt layers| F[huggingface]
```

JAX is fast (because of XLA compilation), works on CUDA and TPU.
This repo implements DiLoCo to use any number of heterogenous compute (kaggle, molab, colab, lightning) without direct connection (only huggingface chekpointing).
