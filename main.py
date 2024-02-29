import torch
import triton
import ipdb
import ref

assert torch.cuda.is_available()
torch.set_default_device('cuda')
torch.set_default_dtype(torch.float32)

TENSOR_FORMAT = 'sbhd'
FUSED = False

S = 4
B = 6
H = 8
D = 10
SHAPE = (S, B, H, D)

S2 = 4
D2 = 6
SHAPE2 = (S2, 1, 1, D2)

def main():
    print('hello')

    t = torch.randn(SHAPE)
    freqs = torch.randn(SHAPE2)
    emb = ref.apply_rotary_pos_emb(t, freqs, tensor_format=TENSOR_FORMAT, fused=FUSED)
    
    ipdb.set_trace()


if __name__ == '__main__':
    main()
