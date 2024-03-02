import torch
import triton
import triton.language as tl
import ipdb
import ref

assert torch.cuda.is_available()
torch.set_default_device('cuda')
torch.set_default_dtype(torch.float32)


TENSOR_FORMAT = 'sbhd'
FUSED = False

S = 10
B = 100
H = 100
D = 1024
SHAPE = (S, B, H, D)
SHAPE_FREQS = (S, 1, 1, D)


def measure_cuda_time(msg, func, *args, **kwargs):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    ret = func(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)

    print(f'{msg}: {elapsed_time_ms:.03f}')
    return ret


def main():
    print('==== hello ====')

    # Random tensors for experimentation
    t = torch.randn(SHAPE, requires_grad=True)
    freqs = torch.randn(SHAPE_FREQS, requires_grad=True)
    grad = torch.randn_like(t)

    # Torch forward
    msg = 'elapsed time torch fw'
    emb = measure_cuda_time(msg, ref.apply_rotary_pos_emb, t, freqs, tensor_format=TENSOR_FORMAT, fused=FUSED)

    # Torch backward
    msg = 'elapsed time torch bw'
    measure_cuda_time(msg, emb.backward, grad)


    # Prepare for triton
    emb_triton = torch.empty_like(emb)
    t_grad_triton = torch.empty_like(t)
    freqs_grad_triton = torch.empty_like(freqs)
    grid = (S, )

    # Triton forward
    msg = 'elapsed time triton fw'
    measure_cuda_time(msg, rope_fw[grid], t, freqs, S, B, H, D, emb_triton)

    # Triton backward
    msg = 'elapsed time triton bw'
    measure_cuda_time(msg, rope_bw[grid], t, freqs, S, B, H, D, grad, t_grad_triton, freqs_grad_triton)

    # Show differences
    diff_emb = emb - emb_triton
    diff_t_grad = t.grad - t_grad_triton
    diff_freqs_grad = freqs.grad - freqs_grad_triton

    diff_emb_max = diff_emb.abs().max().item()
    diff_t_grad_max = diff_t_grad.abs().max().item()
    diff_freqs_grad_max = diff_freqs_grad.abs().max().item()

    print(f'diff_emb_max: {diff_emb_max:.20f}')
    print(f'diff_t_grad_max: {diff_t_grad_max:.20f}')
    print(f'diff_freqs_grad_max: {diff_freqs_grad_max:.20f}')

    print('==== bye ====')



    
@triton.jit
def rope_fw(
    t_ptr, 
    freqs_ptr, 
    S: tl.constexpr, 
    B: tl.constexpr, 
    H: tl.constexpr, 
    D: tl.constexpr, 
    output_ptr,
):
    pid = tl.program_id(0)

    freqs_0 = tl.load(freqs_ptr + pid*D + tl.arange(0, D//2))
    freqs_1 = tl.load(freqs_ptr + pid*D + tl.arange(D//2, D))
    sin_0 = tl.sin(freqs_0)
    sin_1 = tl.sin(freqs_1)
    cos_0 = tl.cos(freqs_0)
    cos_1 = tl.cos(freqs_1)

    for b in range(B):
        for h in range(H):
            offset = pid*B*H*D + b*H*D + h*D
            t_0 = tl.load(t_ptr + offset + tl.arange(0, D//2))
            t_1 = tl.load(t_ptr + offset + tl.arange(D//2, D))

            emb_0 = t_0*cos_0 - t_1*sin_0
            emb_1 = t_1*cos_1 + t_0*sin_1

            tl.store(output_ptr + offset + tl.arange(0, D//2), emb_0)
            tl.store(output_ptr + offset + tl.arange(D//2, D), emb_1)



@triton.jit
def rope_bw(
    t_ptr, 
    freqs_ptr, 
    S: tl.constexpr, 
    B: tl.constexpr, 
    H: tl.constexpr, 
    D: tl.constexpr, 
    ptr_grad,
    t_grad_ptr,
    freqs_grad_ptr,
):
    pid = tl.program_id(0)

    freqs_0 = tl.load(freqs_ptr + pid*D + tl.arange(0, D//2))
    freqs_1 = tl.load(freqs_ptr + pid*D + tl.arange(D//2, D))
    sin_0 = tl.sin(freqs_0)
    sin_1 = tl.sin(freqs_1)
    cos_0 = tl.cos(freqs_0)
    cos_1 = tl.cos(freqs_1)

    sin_0_grad = tl.zeros_like(sin_0)
    sin_1_grad = tl.zeros_like(sin_1)
    cos_0_grad = tl.zeros_like(cos_0)
    cos_1_grad = tl.zeros_like(cos_1)

    for b in range(B):
        for h in range(H):
            offset = pid*B*H*D + b*H*D + h*D
            t_0 = tl.load(t_ptr + offset + tl.arange(0, D//2))
            t_1 = tl.load(t_ptr + offset + tl.arange(D//2, D))

            # emb_0 = t_0*cos_0 - t_1*sin_0
            # emb_1 = t_1*cos_1 + t_0*sin_1

            emb_0_grad = tl.load(ptr_grad + offset + tl.arange(0, D//2))
            emb_1_grad = tl.load(ptr_grad + offset + tl.arange(D//2, D))

            t_0_grad = emb_0_grad*cos_0 + emb_1_grad*sin_1
            t_1_grad = -emb_0_grad*sin_0 + emb_1_grad*cos_1
            tl.store(t_grad_ptr + offset + tl.arange(0, D//2), t_0_grad)
            tl.store(t_grad_ptr + offset + tl.arange(D//2, D), t_1_grad)

            cos_0_grad += t_0 * emb_0_grad
            cos_1_grad += t_1 * emb_1_grad
            sin_0_grad += -t_1 * emb_0_grad
            sin_1_grad += t_0 * emb_1_grad
    
    tl.store(freqs_grad_ptr + pid*D + tl.arange(0, D//2), -sin_0*cos_0_grad + cos_0*sin_0_grad)
    tl.store(freqs_grad_ptr + pid*D + tl.arange(D//2, D), -sin_1*cos_1_grad + cos_1*sin_1_grad)





if __name__ == '__main__':
    main()
