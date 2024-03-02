import torch
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
assert D2%2 == 0
assert D2 <= D <= BLOCKSIZE



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
    cos_0 = cos[..., :d//2]
    cos_1 = cos[..., d//2:]
    sin_0 = sin[..., :d//2]
    sin_1 = sin[..., d//2:]

    t_0 = t[..., :d//2]
    t_1 = t[..., d//2:d]
    t_2 = t[..., d:]

    emb_0 = t_0 * cos_0 - t_1 * sin_0
    emb_1 = t_1 * cos_1 + t_0 * sin_1
    emb_2 = t_2
    emb = np.concatenate([emb_0, emb_1, emb_2], axis=-1)

    # ==== Backward ====
    emb_grad = grad
    emb_0_grad = emb_grad[..., :d//2]
    emb_1_grad = emb_grad[..., d//2:d]
    emb_2_grad = emb_grad[..., d:]

    t_0_grad = emb_0_grad * cos_0 + emb_1_grad * sin_1
    t_1_grad = -emb_0_grad * sin_0 + emb_1_grad * cos_1
    t_2_grad = emb_2_grad
    t_grad = np.concatenate([t_0_grad, t_1_grad, t_2_grad], axis=-1)

    cos_0_grad = (t_0 * emb_0_grad).sum(axis=(1, 2), keepdims=True)
    cos_1_grad = (t_1 * emb_1_grad).sum(axis=(1, 2), keepdims=True)
    sin_0_grad = (-t_1 * emb_0_grad).sum(axis=(1, 2), keepdims=True)
    sin_1_grad = (t_0 * emb_1_grad).sum(axis=(1, 2), keepdims=True)

    cos_grad = np.concatenate([cos_0_grad, cos_1_grad], axis=-1)
    sin_grad = np.concatenate([sin_0_grad, sin_1_grad], axis=-1)
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

    print(f'diff_emb_max: {diff_emb_max:.20f}')
    print(f'diff_t_grad_max: {diff_t_grad_max:.20f}')
    print(f'diff_freqs_grad_max: {diff_freqs_grad_max:.20f}')


if __name__ == '__main__':
    test_rope_numpy()
