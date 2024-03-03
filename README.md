# Rotary Position Embedding
A Triton implementation mimicking the
[pytorch one from NVIDIA](https://github.com/NVIDIA/TransformerEngine/blob/b8eea8aaa94bb566c3a12384eda064bda8ac4fd7/transformer_engine/pytorch/attention.py#L1170-L1230).


## Environment
```
docker pull donghyunismyname/rope:1.0
```


## Experiment
Ran on NVIDIA RTX 2080 (8GB VRAM)
```
==== hello ====
elapsed time torch fw: 79.942 ms
elapsed time torch bw: 149.482 ms
elapsed time triton fw: 5.280 ms
elapsed time triton bw: 25.857 ms
diff_emb_max: 0.00000047683715820312
diff_t_grad_max: 0.00000047683715820312
diff_freqs_grad_max: 0.00062561035156250000
==== bye ====
```

## Run
```
python main.py
```
