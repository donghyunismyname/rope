import torch
import triton
import triton.language as tl
import numpy as np
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
    '''
    Forward and backward pass of the RoPE operation in numpy.
    '''
    # ==== Type Conversion ====
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
    emb_grad = grad
    lincomb_grad = emb_grad[..., :d]
    t_pass_grad = emb_grad[..., d:]
    t_cut_grad = lincomb_grad * cos
    t_cut_rot_grad = lincomb_grad * sin
    cos_grad = (t_cut * lincomb_grad).sum(axis=(1, 2), keepdims=True)
    sin_grad = (t_cut_rot * lincomb_grad).sum(axis=(1, 2), keepdims=True)

    t_cut_grad[..., :h] += t_cut_rot_grad[..., h:]
    t_cut_grad[..., h:] += -t_cut_rot_grad[..., :h]
    t_grad = np.concatenate([t_cut_grad, t_pass_grad], axis=-1)

    freqs_cut_grad = -sin*cos_grad + cos*sin_grad
    freqs_grad = np.zeros_like(freqs)
    freqs_grad[:t.shape[0]] = freqs_cut_grad

    # ==== Type Conversion ==== 
    emb = torch.tensor(emb)
    t_grad = torch.tensor(t_grad)
    freqs_grad = torch.tensor(freqs_grad)

    return emb, t_grad, freqs_grad


def test_rope_numpy():
    t = torch.randn(SHAPE, requires_grad=True)
    freqs = torch.randn(SHAPE2, requires_grad=True)
    emb = ref.apply_rotary_pos_emb(t, freqs, tensor_format=TENSOR_FORMAT, fused=FUSED)
    grad = torch.randn_like(emb)
    emb.backward(grad)

    np_emb, np_t_grad, np_freqs_grad = rope_numpy(t, freqs, grad)

    diff_emb = emb - np_emb
    diff_t_grad = t.grad - np_t_grad
    diff_freqs_grad = freqs.grad - np_freqs_grad

    diff_emb_max = diff_emb.abs().max().item()
    diff_t_grad_max = diff_t_grad.abs().max().item()
    diff_freqs_grad_max = diff_freqs_grad.abs().max().item()

    print(f'diff_emb_max: {diff_emb_max:.20f}');
    print(f'diff_t_grad_max: {diff_t_grad_max:.20f}');
    print(f'diff_freqs_grad_max: {diff_freqs_grad_max:.20f}');





    
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
