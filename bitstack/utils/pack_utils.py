import torch
import triton
import triton.language as tl

def pack_sign(w_sign):
    assert (w_sign.abs() == 1).all()
    binary = ((w_sign + 1) // 2).view(-1)
    n = binary.numel()
    n_padding = (8 - n % 8) % 8
    if n_padding > 0:
        print(f"Padding {n_padding} bits")
        binary = torch.cat([binary, torch.zeros(n_padding, dtype=binary.dtype, device=binary.device)])

    binary = binary.view(-1, 8)
    packed = torch.zeros(binary.size(0), dtype=torch.uint8, device=binary.device)
    for i in range(8):
        packed |=(binary[:, i] << (7 - i))
    return packed

def unpack_sign(packed, original_shape):
    bit_masks = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], device=packed.device, dtype=torch.uint8)
    unpacked  = ((packed.unsqueeze(-1) & bit_masks) > 0).to(torch.int8).view(-1)
    n = original_shape.numel()
    unpacked = unpacked[:n].view(original_shape)
    unpacked = unpacked * 2 - 1
    return unpacked

# Cache for bit masks to avoid recreating them on every function call
_bit_masks_cache = {}

def get_bit_masks(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Retrieves cached bit masks or creates them if not present.
    
    Args:
        device (torch.device): The device on which the bit masks should reside.
        dtype (torch.dtype): The data type of the bit masks.
        
    Returns:
        torch.Tensor: A tensor containing the bit masks.
    """
    global _bit_masks_cache
    key = (device, dtype)
    if key not in _bit_masks_cache:
        # Bit masks for each bit in a byte (from MSB to LSB)
        _bit_masks_cache[key] = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], 
                                            device=device, dtype=dtype)
    return _bit_masks_cache[key]

def unpack_sign_v2(packed: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """
    Unpacks a packed byte tensor into a tensor of -1 and 1 based on the bits.
    
    Args:
        packed (torch.Tensor): A 1D byte tensor containing packed bits.
        original_shape (torch.Size): The desired shape of the output tensor.
        
    Returns:
        torch.Tensor: A tensor of shape `original_shape` with values -1 and 1.
    """
    if packed.dtype != torch.uint8:
        raise ValueError(f"Expected packed tensor of dtype torch.uint8, but got {packed.dtype}")
    
    bit_masks = get_bit_masks(packed.device, packed.dtype)
    
    # Perform bitwise AND with each bit mask to extract individual bits
    # Shape: (num_bytes, 8)
    bits = (packed.unsqueeze(-1) & bit_masks) > 0  # Boolean tensor
    
    # Convert boolean to int8 (-1 and 1)
    # First convert to int8: False -> 0, True -> 1
    bits_int = bits.to(torch.int8)
    
    # Map 0 to -1 and 1 to 1
    unpacked = bits_int * 2 - 1  # Now contains -1 and 1
    
    # Flatten and select the first n elements
    n = original_shape.numel()
    unpacked = unpacked.view(-1)[:n]
    
    # Reshape to the original desired shape
    return unpacked.view(original_shape)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'PACKED_BLOCK_SIZE': 16}),
        triton.Config({'BLOCK_SIZE': 256, 'PACKED_BLOCK_SIZE': 32}),
        triton.Config({'BLOCK_SIZE': 512, 'PACKED_BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 1024, 'PACKED_BLOCK_SIZE': 128}),
    ],
    key=['n_elements'],
)
@triton.jit
def unpack_sign_kernel(
    packed_ptr,     # Pointer to the packed input tensor
    unpacked_ptr,   # Pointer to the output tensor
    n_elements,     # Total number of elements to unpack
    BLOCK_SIZE: tl.constexpr,
    PACKED_BLOCK_SIZE: tl.constexpr
):

    # Program ID and block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    packed_start = pid * PACKED_BLOCK_SIZE

    # Generate packed byte indices for the block
    packed_offsets = packed_start + tl.arange(0, PACKED_BLOCK_SIZE) # Shape: (PACKED_BLOCK_SIZE,)
    # Mask to ensure we don't read beyond the packed data
    packed_mask = packed_offsets < ((n_elements + 7) // 8) # Shape: (PACKED_BLOCK_SIZE,)

    # Load all packed bytes for this block once
    packed = tl.load(packed_ptr + packed_offsets, mask=packed_mask, other=0) # Shape: (PACKED_BLOCK_SIZE, )

    # Compute the bit positions within each byte
    bit_offsets = tl.arange(0, 8) # Shape: (8,)
    # Expand packed bytes to align with bit positions
    packed = packed[:, None]  # Shape: (PACKED_BLOCK_SIZE, 1)

    # Extract bits: shift and mask
    bits = (packed >> (7 - bit_offsets)) & 1  # Shape: (PACKED_BLOCK_SIZE, 8)

    # Convert bits to signs (-1 or +1)
    signs = bits * 2 - 1  # Shape: (PACKED_BLOCK_SIZE, 8)

    # Compute global element indices
    element_offsets = block_start + (tl.arange(0, PACKED_BLOCK_SIZE)[:, None] * 8 + bit_offsets)
    element_offsets = tl.reshape(element_offsets, [BLOCK_SIZE])  # Shape: (PACKED_BLOCK_SIZE * 8,)

    # Flatten signs for storage
    signs = tl.reshape(signs, [BLOCK_SIZE])

    # Create a mask to prevent out-of-bounds writes
    mask = element_offsets < n_elements

    # Store the unpacked signs as float16
    tl.store(unpacked_ptr + element_offsets, signs.to(tl.float16), mask=mask)


def unpack_sign_triton(packed, original_shape):
    n_elements = original_shape.numel()
    unpacked = torch.empty(n_elements, dtype=torch.int8, device=packed.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    unpack_sign_kernel[grid](
        packed, unpacked,
        n_elements,
    )

    unpacked = unpacked.view(original_shape)
    return unpacked
