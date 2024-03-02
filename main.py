import torch
import triton
import triton.language as tl
import ipdb
import ref

assert torch.cuda.is_available()
torch.set_default_device('cuda')
torch.set_default_dtype(torch.float32)


BLOCKSIZE = 1024

TENSOR_FORMAT = 'sbhd'
FUSED = False

S = 4
B = 6
H = 8
D = 1024
SHAPE = (S, B, H, D)

S2 = S + 10
D2 = max(10, D - 10)
SHAPE2 = (S2, 1, 1, D2)

assert S2 >= S
assert D%2 == 0
assert D2%2 == 0
assert D2 <= D <= BLOCKSIZE


def main():
    print('hello')

    t = torch.randn(SHAPE, requires_grad=True)
    freqs = torch.randn(SHAPE2, requires_grad=True)
    emb = ref.apply_rotary_pos_emb(t, freqs, tensor_format=TENSOR_FORMAT, fused=FUSED)
    grad = torch.randn_like(emb)
    emb.backward(grad)

    emb_rope = torch.empty_like(emb)
    grid = (S, )
    rope_fw[grid](t, S, B, H, D, freqs, S2, D2, emb_rope)

    diff = emb - emb_rope
    
    print('bye')



    
@triton.jit
def rope_fw(
    t_ptr, 
    S: tl.constexpr, 
    B: tl.constexpr, 
    H: tl.constexpr, 
    D: tl.constexpr, 
    freqs_ptr, 
    S2: tl.constexpr, 
    D2: tl.constexpr,
    output_ptr,
):
    pid = tl.program_id(0)

    freqs_0 = tl.load(freqs_ptr + pid*D2 + tl.arange(0, D2//2))
    freqs_1 = tl.load(freqs_ptr + pid*D2 + tl.arange(D2//2, D2))
    sin_0 = tl.sin(freqs_0)
    sin_1 = tl.sin(freqs_1)
    cos_0 = tl.cos(freqs_0)
    cos_1 = tl.cos(freqs_1)

    for b in range(B):
        for h in range(H):
            offset = pid*B*H*D + b*H*D + h*D
            t_0 = tl.load(t_ptr + offset + tl.arange(0, D2//2))
            t_1 = tl.load(t_ptr + offset + tl.arange(D2//2, D2))
            t_2 = tl.load(t_ptr + offset + tl.arange(D2, D))







@triton.jit
def rope_bw(t: torch.Tensor, freqs: torch.Tensor, emb: torch.Tensor, grad: torch.Tensor):
    pass



if __name__ == '__main__':
    main()
