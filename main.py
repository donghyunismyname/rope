import torch
import triton
import numpy as np
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
    
    ipdb.set_trace()
    print('bye')



def rope_numpy(t, freqs, grad):
    t = t.detach().cpu().numpy()
    freqs = freqs.detach().cpu().numpy()
    grad = grad.detach().cpu().numpy()

    # ==== Forward ====
    freqs_cut = freqs[:t.shape[0]]
    cos = np.cos(freqs_cut)
    sin = np.sin(freqs_cut)

    d = freqs_cut.shape[-1]
    t_cut, t_pass = t[..., :d], t[..., d:]

    h = t_cut.shape[-1] // 2
    t0 = t_cut[..., :h]
    t1 = t_cut[..., h:]
    t_cut_rot = np.concatenate([-t1, t0], axis=-1)

    lincomb = t_cut * cos + t_cut_rot * sin
    emb = np.concatenate([lincomb, t_pass], axis=-1)

    # ==== Backward ====
    # emb_grad = grad
    # t_grad = emb_grad * cos
    # tt_grad = emb_grad * sin

    

    
    emb = torch.tensor(emb)

    return emb


def test_rope_numpy():
    t = torch.randn(SHAPE, requires_grad=True)
    freqs = torch.randn(SHAPE2, requires_grad=True)
    emb = ref.apply_rotary_pos_emb(t, freqs, tensor_format=TENSOR_FORMAT, fused=FUSED)
    grad = torch.randn_like(emb)
    emb.backward(grad)

    emb_numpy = rope_numpy(t, freqs, grad)

    diff_emb = emb - emb_numpy
    print(f'diff: {diff_emb.abs().max().item():.10f}');

    ipdb.set_trace()




    
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
    idx = tl.arange(0, D)

    offset = pid * D2
    freqs = tl.load(
        freqs_ptr + offset + idx,
        mask=idx < D2,
        other=0.0,
    )
    sin = tl.sin(freqs)
    cos = tl.cos(freqs)

    for b in range(B):
        for h in range(H):
            # -1, 1, -1, 1, ...
            coef = (idx % 2) * 2 - 1

            # 1, 0, 3, 2, 5, 4, ...
            idx_swap = (idx // 2 * 2) + (idx + 1) % 2
            
            offset = pid * B * H * D + b * H * D + h * D
            t = tl.load(t_ptr + offset + idx)
            t0 = tl.load(t_ptr + offset + tl.arange(0, D//2) * 2 + 0)
            t1 = tl.load(t_ptr + offset + tl.arange(0, D//2) * 2 + 1)

            t_swap = tl.load(t_ptr + offset + idx_swap)

            # result = t * cos + t_swap * coef * sin
            # result = t * cos
            result = t_swap * coef
            tl.store(output_ptr + offset + idx, result)






@triton.jit
def rope_bw(t: torch.Tensor, freqs: torch.Tensor, emb: torch.Tensor, grad: torch.Tensor):
    pass



if __name__ == '__main__':
    test_rope_numpy()
    # main()
