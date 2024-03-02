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

def main():
    print('hello')

    # Events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Random tensors for experimentation
    t = torch.randn(SHAPE, requires_grad=True)
    freqs = torch.randn(SHAPE_FREQS, requires_grad=True)
    grad = torch.randn_like(t)

    # Torch forward
    start_event.record()
    emb = ref.apply_rotary_pos_emb(t, freqs, tensor_format=TENSOR_FORMAT, fused=FUSED)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f'elapsed time torch fw: {elapsed_time_ms:.03f}')

    # Torch backward
    start_event.record()
    emb.backward(grad)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f'elapsed time torch bw: {elapsed_time_ms:.03f}')



    emb_triton = torch.empty_like(emb)
    grid = (S, )

    start_event.record()
    rope_fw[grid](t, freqs, S, B, H, D, emb_triton)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f'elapsed time triton fw: {elapsed_time_ms:.03f}')

    diff = emb - emb_triton
    diff_max = diff.abs().max().item()
    print(f'diff_max: {diff_max:.20f}')
    print('bye')



    
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
def rope_bw(t: torch.Tensor, freqs: torch.Tensor, emb: torch.Tensor, grad: torch.Tensor):
    pass



if __name__ == '__main__':
    main()
