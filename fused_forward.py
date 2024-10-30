import torch

import triton
import triton.language as tl

from utils.pack_utils import pack_sign

def get_cuda_autotune_config():


    BLOCK_SIZE_M_OPTIONS = [16, 32, 64]
    BLOCK_SIZE_N_OPTIONS = [16, 32, 64]


    BLOCK_SIZE_K_OPTIONS = [16]


    GROUP_SIZE_M_OPTIONS = [4, 8]


    NUM_WARPS_OPTIONS = [2, 4, 8]
    NUM_STAGES_OPTIONS = [2, 3, 4]

    configs = []
    for bs_m in BLOCK_SIZE_M_OPTIONS:
        for bs_n in BLOCK_SIZE_N_OPTIONS:
            for bs_k in BLOCK_SIZE_K_OPTIONS:
                for gs_m in GROUP_SIZE_M_OPTIONS:
                    for nw in NUM_WARPS_OPTIONS:
                        for ns in NUM_STAGES_OPTIONS:
                            config = triton.Config({
                                'BLOCK_SIZE_M': bs_m,
                                'BLOCK_SIZE_N': bs_n,
                                'BLOCK_SIZE_K': bs_k,
                                'GROUP_SIZE_M': gs_m,
                            }, num_warps=nw, num_stages=ns)
                            configs.append(config)

    return configs


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['N_ITER', 'M', 'N', 'K'],
)
@triton.jit
def fused_reconstruct_kernel(
    sign_ptr, u_ptr, vt_ptr, output_ptr,
    N_ITER, M, N, K,
    stride_sign_iter, stride_sign_m, stride_sign_n,
    stride_u_iter, stride_u_m, stride_u_k,
    stride_vt_iter, stride_vt_k, stride_vt_n,
    stride_output_m, stride_output_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    """Kernel for computing (sign * u @ vt).sum(dim=0).
    sign: (N_ITER, M, N)
    u: (N_ITER, M, K)
    vt: (N_ITER, K, N)
    output: (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offsets_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offsets_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for n_iter in range(N_ITER):
        sign_ptrs = sign_ptr + (
            n_iter * stride_sign_iter 
            + offsets_m[:, None] * stride_sign_m 
            + offsets_n[None, :] * stride_sign_n
        )
        u_ptrs = u_ptr + (
            n_iter * stride_u_iter
            + offsets_m[:, None] * stride_u_m
        )
        vt_ptrs = vt_ptr + (
            n_iter * stride_vt_iter
            + offsets_n[None, :] * stride_vt_n
        )
        
        iter_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            u_block_ptrs = u_ptrs + (offsets_k[None, :] * stride_u_k)
            vt_block_ptrs = vt_ptrs + (offsets_k[:, None] * stride_vt_k)
            
            u = tl.load(u_block_ptrs, mask=offsets_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            vt = tl.load(vt_block_ptrs, mask=offsets_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            
            iter_acc += tl.dot(u, vt, out_dtype=tl.float32)
            
        sign = tl.load(sign_ptrs, mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N), other=0.0)
        acc += sign * iter_acc
        # acc += tl.where(sign == 1, iter_acc, -iter_acc)

    output = acc.to(tl.float16)

    offsets_output_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_output_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = output_ptr + stride_output_m * offsets_output_m[:, None] + stride_output_n * offsets_output_n[None, :]
    output_mask = (offsets_output_m[:, None] < M) & (offsets_output_n[None, :] < N)
    tl.store(output_ptrs, output, mask=output_mask)


def reconstruct_w_triton(sign, u, vt):
    # This implements the fused computation for (sign * u @ vt).sum(dim=0)
    N_ITER, M, N = sign.shape
    K = u.shape[-1]
    output = torch.empty((M, N), device=u.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    fused_reconstruct_kernel[grid](
        sign, u, vt, output,
        N_ITER, M, N, K,
        sign.stride(0), sign.stride(1), sign.stride(2),
        u.stride(0), u.stride(1), u.stride(2),
        vt.stride(0), vt.stride(1), vt.stride(2),
        output.stride(0), output.stride(1),

    )
    return output





# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16}, num_stages=2, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 16}, num_stages=3, num_warps=8),
#         # Add more configurations as needed
#     ],
#     key=['N_ITER', 'M', 'N', 'K'],
# )
@triton.jit
def fused_reconstruct_and_forward_kernel(
    sign_ptr, u_ptr, vt_ptr, output_ptr,
    x_ptr,
    N_ITER, M, N, K,
    BZ, L,
    stride_sign_iter, stride_sign_m, stride_sign_n,
    stride_u_iter, stride_u_m, stride_u_k,
    stride_vt_iter, stride_vt_k, stride_vt_n,
    stride_x_bz, stride_x_l, stride_x_n,
    stride_output_bz, stride_output_l, stride_output_m,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    """Kernel for computing (sign * u @ vt).sum(dim=0) followed by x @ w.T.
    sign: (N_ITER, M, N)
    u: (N_ITER, M, K)
    vt: (N_ITER, K, N)
    x: (BZ, L, N)
    output: (BZ, L, M)
    """
    # Program IDs for M and batch-sequence dimensions
    pid_m = tl.program_id(axis=0)
    pid_bzl = tl.program_id(axis=1)
    bz = pid_bzl // L
    l = pid_bzl % L

    # Offsets for M and K dimensions
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator for output
    output_vals = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # Loop over N in chunks
    for n_off in range(0, N, BLOCK_SIZE_N):
        offsets_n = n_off + tl.arange(0, BLOCK_SIZE_N)
        n_mask = offsets_n < N

        # Compute w_T_chunk (BLOCK_SIZE_N x BLOCK_SIZE_M)
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
        for n_iter in range(N_ITER):
            sign_ptrs = sign_ptr + (
                n_iter * stride_sign_iter 
                + offsets_m[:, None] * stride_sign_m 
                + offsets_n[None, :] * stride_sign_n
            )
            u_ptrs = u_ptr + (
                n_iter * stride_u_iter
                + offsets_m[:, None] * stride_u_m
            )
            vt_ptrs = vt_ptr + (
                n_iter * stride_vt_iter
                + offsets_n[None, :] * stride_vt_n
            )

            iter_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
            for k in range(0, K, BLOCK_SIZE_K):
                u_block_ptrs = u_ptrs + (offsets_k[None, :] + k) * stride_u_k
                vt_block_ptrs = vt_ptrs + (offsets_k[:, None] + k) * stride_vt_k

                k_mask = offsets_k + k < K
                u = tl.load(u_block_ptrs, mask=k_mask[None, :], other=0.0)
                vt = tl.load(vt_block_ptrs, mask=k_mask[:, None], other=0.0)

                iter_acc += tl.dot(u, vt, out_dtype=tl.float16)

            sign = tl.load(sign_ptrs, mask=(offsets_m[:, None] < M) & n_mask[None, :], other=0.0)
            acc += sign * iter_acc

        # Load x values for the current batch and sequence position
        x_ptrs = x_ptr + bz * stride_x_bz + l * stride_x_l + offsets_n * stride_x_n
        x_vals = tl.load(x_ptrs, mask=n_mask, other=0.0).to(tl.float16)  # Shape: (BLOCK_SIZE_N,)

        # Compute partial output
        output_vals += tl.sum(acc * x_vals[None, :], axis=1)

    # Store the computed output
    output_ptrs = output_ptr + bz * stride_output_bz + l * stride_output_l + offsets_m * stride_output_m
    tl.store(output_ptrs, output_vals, mask=offsets_m < M)

def reconstruct_and_forward_triton(sign, u, vt, x):
    """
    Fused computation for w = (sign * u @ vt).sum(dim=0) followed by matmul(x, w.T)
    Args:
        sign: (N_ITER, M, N)
        u: (N_ITER, M, K)
        vt: (N_ITER, K, N)
        x: (BZ, L, N)
    Returns:
        output: (BZ, L, M)
    """
    N_ITER, M, N = sign.shape
    K = u.shape[-1]
    BZ, L, _ = x.shape
    output = torch.empty((BZ, L, M), device=x.device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        BZ * L
    )
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 16
    NUM_WARPS = 8
    NUM_STAGES = 2
    fused_reconstruct_and_forward_kernel[grid](
        sign, u, vt, output,
        x,
        N_ITER, M, N, K,
        BZ, L,
        sign.stride(0), sign.stride(1), sign.stride(2),
        u.stride(0), u.stride(1), u.stride(2),
        vt.stride(0), vt.stride(1), vt.stride(2),
        x.stride(0), x.stride(1), x.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        NUM_WARPS, NUM_STAGES
    )
    return output

@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['N_ITER', 'M', 'N', 'K'],
)
@triton.jit
def fused_unpack_and_reconstruct_kernel(
    packed_sign_ptr, u_ptr, vt_ptr, output_ptr,
    N_ITER, M, N, K,
    n_sign_elements,
    stride_u_iter, stride_u_m, stride_u_k,
    stride_vt_iter, stride_vt_k, stride_vt_n,
    stride_output_m, stride_output_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    """Kernel for computing (sign * u @ vt).sum(dim=0) with sign unpacking fused."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)


    # Pre-compute constant offsets outside the n_iter loop
    base_m = offsets_m[:, None] * N  # Shape: (BLOCK_SIZE_M, 1)
    base_n = offsets_n[None, :]      # Shape: (1, BLOCK_SIZE_N)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for n_iter in range(N_ITER):
        # Compute byte indices directly
        element_indices = n_iter * M * N + base_m + base_n  # Shape: (BLOCK_SIZE_M, BLOCK_SIZE_N)
        element_indices = tl.reshape(element_indices, (BLOCK_SIZE_M * BLOCK_SIZE_N,))  # Shape: (BLOCK_SIZE_M * BLOCK_SIZE_N,)

        byte_indices = element_indices // 8
        bit_indices = element_indices % 8

        # Load the packed bytes
        byte_ptrs = packed_sign_ptr + byte_indices
        byte_mask = byte_indices < ((n_sign_elements + 7) // 8)
        packed_bytes = tl.load(byte_ptrs, mask=byte_mask, other=0)

        # Extract signs
        bits = (packed_bytes >> (7 - bit_indices)) & 1
        signs = bits * 2 - 1
        signs = tl.reshape(signs, (BLOCK_SIZE_M, BLOCK_SIZE_N))

        u_ptrs = u_ptr + (
            n_iter * stride_u_iter
            + offsets_m[:, None] * stride_u_m
        )
        vt_ptrs = vt_ptr + (
            n_iter * stride_vt_iter
            + offsets_n[None, :] * stride_vt_n
        )

        iter_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            u_block_ptrs = u_ptrs + (offsets_k[None, :] * stride_u_k)
            vt_block_ptrs = vt_ptrs + (offsets_k[:, None] * stride_vt_k)

            u = tl.load(u_block_ptrs, mask=offsets_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            vt = tl.load(vt_block_ptrs, mask=offsets_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

            iter_acc += tl.dot(u, vt, out_dtype=tl.float32)

        acc += signs * iter_acc
        # acc += tl.where(signs == 1, iter_acc, -iter_acc)

    output = acc.to(tl.float16)

    offsets_output_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_output_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = output_ptr + stride_output_m * offsets_output_m[:, None] + stride_output_n * offsets_output_n[None, :]
    output_mask = (offsets_output_m[:, None] < M) & (offsets_output_n[None, :] < N)
    tl.store(output_ptrs, output, mask=output_mask)

def unpack_and_reconstruct_w_triton(packed_sign, u, vt):
    # Extract dimensions
    N_ITER, M, K = u.shape
    N = vt.shape[2]
    output = torch.empty((M, N), device=u.device, dtype=torch.float16)
    n_sign_elements = N_ITER * M * N
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    fused_unpack_and_reconstruct_kernel[grid](
        packed_sign, u, vt, output,
        N_ITER, M, N, K,
        n_sign_elements,
        u.stride(0), u.stride(1), u.stride(2),
        vt.stride(0), vt.stride(1), vt.stride(2),
        output.stride(0), output.stride(1),

    )
    return output

@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['N_ITER', 'M', 'N', 'K'],
    reset_to_zero=['output_ptr']
)
@triton.jit
def fused_unpack_and_reconstruct_kernel_v2(
    packed_sign_ptr, u_ptr, vt_ptr, output_ptr,
    N_ITER, M, N, K,
    n_sign_elements,
    stride_packed_sign_iter, stride_packed_sign_m, stride_packed_sign_n,
    stride_u_iter, stride_u_m, stride_u_k,
    stride_vt_iter, stride_vt_k, stride_vt_n,
    stride_output_m, stride_output_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,

):
    # First program ID for spatial dimensions (M, N)
    pid_spatial = tl.program_id(axis=0)
    # Second program ID for temporal dimension (N_ITER)
    pid_iter = tl.program_id(axis=1)
    # Calculate spatial indices
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_spatial // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    local_pid = pid_spatial % num_pid_in_group
    pid_n = local_pid // group_size_m
    pid_m = first_pid_m + local_pid % group_size_m

    # Calculate offsets
    offsets_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None]
    offsets_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :]
    offsets_k = tl.arange(0, BLOCK_SIZE_K)

    # Unpack signs
    # PACKED_BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE_N // 8

    # offsets_n_packed = pid_n * PACKED_BLOCK_SIZE_N + tl.arange(0, PACKED_BLOCK_SIZE_N)
    # packed_sign_ptrs = packed_sign_ptr + (
    #     pid_iter * stride_packed_sign_iter +
    #     offsets_m * stride_packed_sign_m +
    #     offsets_n_packed * stride_packed_sign_n
    # )
    # packed_bytes = tl.load(packed_sign_ptrs)
    # bit_offsets = tl.arange(0, 8) # Shape: (8,)
    # packed_bytes = packed_bytes[:, :, None] # Shape: (BLOCK_SIZE_M, PACKED_BLOCK_SIZE_N, 1)
    
    element_indices = pid_iter * M * N + offsets_m * N + offsets_n
    element_indices = tl.reshape(element_indices, (BLOCK_SIZE_M * BLOCK_SIZE_N,))
    byte_indices = element_indices // 8
    bit_indices = element_indices % 8
    byte_ptrs = packed_sign_ptr + byte_indices
    byte_mask = byte_indices < ((n_sign_elements + 7) // 8)
    packed_bytes = tl.load(byte_ptrs, mask=byte_mask, other=0)
    
    bits = (packed_bytes >> (7 - bit_indices)) & 1 # Shape: (BLOCK_SIZE_M, PACKED_BLOCK_SIZE_N, 8)
    signs = bits * 2 - 1
    signs = tl.reshape(signs, (BLOCK_SIZE_M, BLOCK_SIZE_N))
    # Load and multiply matrices
    u_ptrs = u_ptr + pid_iter * stride_u_iter + offsets_m * stride_u_m
    vt_ptrs = vt_ptr + pid_iter * stride_vt_iter + offsets_n * stride_vt_n
    # Initialize accumulator
    iter_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        u_block_ptrs = u_ptrs + (offsets_k[None, :] * stride_u_k)
        vt_block_ptrs = vt_ptrs + (offsets_k[:, None] * stride_vt_k)

        u = tl.load(u_block_ptrs, mask=offsets_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        vt = tl.load(vt_block_ptrs, mask=offsets_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        iter_acc += tl.dot(u, vt, out_dtype=tl.float32)

    
    # Apply signs with proper masking
    # iter_acc = signs * iter_acc
    
    # Convert to output dtype
    output = (signs * iter_acc).to(tl.float16)
    
    offsets_output_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_output_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = output_ptr + stride_output_m * offsets_output_m[:, None] + stride_output_n * offsets_output_n[None, :]
    output_mask = (offsets_output_m[:, None] < M) & (offsets_output_n[None, :] < N)

    # Atomic accumulate with proper masking
    tl.atomic_add(output_ptrs, output, mask=output_mask)

def unpack_and_reconstruct_w_triton_v2(packed_sign, u, vt):
    N_ITER, M, K = u.shape
    N = vt.shape[2]
    if len(packed_sign.shape) == 2:
        packed_sign = packed_sign.view(N_ITER, M, -1)
    output = torch.zeros((M, N), device=u.device, dtype=torch.float16)
    n_sign_elements = N_ITER * M * N
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        N_ITER,
    )

    fused_unpack_and_reconstruct_kernel_v2[grid](
        packed_sign, u, vt, output,
        N_ITER, M, N, K,
        n_sign_elements,
        packed_sign.stride(0), packed_sign.stride(1), packed_sign.stride(2),
        u.stride(0), u.stride(1), u.stride(2),
        vt.stride(0), vt.stride(1), vt.stride(2),
        output.stride(0), output.stride(1),

    )
    return output


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['N_ITER', 'M', 'N', 'K'],
    reset_to_zero=['output_ptr']
)
@triton.jit
def fused_unpack_and_reconstruct_kernel_v3(
    packed_sign_ptr, u_ptr, vt_ptr, output_ptr,
    N_ITER, M, N, K,
    n_sign_elements,
    stride_packed_sign_iter, stride_packed_sign_m, stride_packed_sign_n,
    stride_u_iter, stride_u_m, stride_u_k,
    stride_vt_iter, stride_vt_k, stride_vt_n,
    stride_output_iter, stride_output_m, stride_output_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    # First program ID for spatial dimensions (M, N)
    pid_spatial = tl.program_id(axis=0)
    # Second program ID for temporal dimension (N_ITER)
    pid_iter = tl.program_id(axis=1)
    # Calculate spatial indices
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_spatial // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    local_pid = pid_spatial % num_pid_in_group
    pid_n = local_pid // group_size_m
    pid_m = first_pid_m + local_pid % group_size_m

    # Calculate offsets
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)

    # Unpack signs
    PACKED_BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE_N // 8
    offsets_n_packed = pid_n * PACKED_BLOCK_SIZE_N + tl.arange(0, PACKED_BLOCK_SIZE_N)
    packed_sign_ptrs = packed_sign_ptr + (
        pid_iter * stride_packed_sign_iter +
        offsets_m[:, None] * stride_packed_sign_m +
        offsets_n_packed[None, :] * stride_packed_sign_n
    )
    packed_bytes = tl.load(packed_sign_ptrs)

    bit_offsets = tl.arange(0, 8) # Shape: (8,)
    packed_bytes = packed_bytes[:, :, None] # Shape: (BLOCK_SIZE_M, PACKED_BLOCK_SIZE_N, 1)

    bits = (packed_bytes >> (7 - bit_offsets)) & 1 # Shape: (BLOCK_SIZE_M, PACKED_BLOCK_SIZE_N, 8)
    signs = bits * 2 - 1
    signs = tl.reshape(signs, (BLOCK_SIZE_M, BLOCK_SIZE_N))
    # Load and multiply matrices
    u_ptrs = u_ptr + pid_iter * stride_u_iter + offsets_m[:, None] * stride_u_m
    vt_ptrs = vt_ptr + pid_iter * stride_vt_iter + offsets_n[None, :] * stride_vt_n
    # Initialize accumulator
    iter_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        u_block_ptrs = u_ptrs + (offsets_k[None, :] * stride_u_k)
        vt_block_ptrs = vt_ptrs + (offsets_k[:, None] * stride_vt_k)

        u = tl.load(u_block_ptrs, mask=offsets_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        vt = tl.load(vt_block_ptrs, mask=offsets_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        iter_acc += tl.dot(u, vt, out_dtype=tl.float32)

    
    # Apply signs with proper masking
    # iter_acc = signs * iter_acc
    
    # Convert to output dtype
    output = (signs * iter_acc).to(tl.float16)
    
    offsets_output_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_output_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = output_ptr + pid_iter * stride_output_iter + stride_output_m * offsets_output_m[:, None] + stride_output_n * offsets_output_n[None, :]
    output_mask = (offsets_output_m[:, None] < M) & (offsets_output_n[None, :] < N)

    # Atomic accumulate with proper masking
    tl.atomic_add(output_ptrs, output, mask=output_mask)

def unpack_and_reconstruct_w_triton_v3(packed_sign, u, vt):
    N_ITER, M, K = u.shape
    N = vt.shape[2]
    if len(packed_sign.shape) == 2:
        packed_sign = packed_sign.view(N_ITER, M, -1)
    output = torch.zeros((N_ITER, M, N), device=u.device, dtype=torch.float16)
    n_sign_elements = N_ITER * M * N
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        N_ITER,
    )

    fused_unpack_and_reconstruct_kernel_v3[grid](
        packed_sign, u, vt, output,
        N_ITER, M, N, K,
        n_sign_elements,
        packed_sign.stride(0), packed_sign.stride(1), packed_sign.stride(2),
        u.stride(0), u.stride(1), u.stride(2),
        vt.stride(0), vt.stride(1), vt.stride(2),
        output.stride(0), output.stride(1), output.stride(2),

    )
    output = output.sum(dim=0)
    return output

if __name__ == "__main__":
    torch.manual_seed(0)
    # a = torch.randn((4096, 16), device='cuda', dtype=torch.float16)
    # b = torch.randn((16, 4096), device='cuda', dtype=torch.float16)
    # triton_output = matmul(a, b)
    # torch_output = torch.matmul(a, b)
    M = 32
    N = 32
    K = 16
    N_ITER = 1
    x = torch.randn((1, 1, N), device='cuda', dtype=torch.float16)
    sign = torch.sign(torch.randn((N_ITER, M, N), device='cuda', dtype=torch.float16))
    sign[sign == 0] = 1
    packed_sign = pack_sign(sign.to(torch.int8)).reshape(N_ITER, M, -1)
    u = torch.randn((N_ITER, M, K), device='cuda', dtype=torch.float16)
    vt = torch.randn((N_ITER, K, N), device='cuda', dtype=torch.float16)
    # triton_output = reconstruct_w_triton(sign, u, vt)
    # torch_output = (sign * torch.matmul(u, vt)).sum(dim=0)
    # triton_output = reconstruct_and_forward_triton(sign, u, vt, x)
    # torch_output = x @ ((sign * torch.matmul(u, vt)).sum(dim=0)).T
    triton_output = unpack_and_reconstruct_w_triton_v2(packed_sign, u, vt)
    torch_output = (sign * torch.matmul(u, vt)).sum(dim=0)
    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
        mse = (triton_output - torch_output).pow(2).mean()
        print(f"MSE={mse}")
