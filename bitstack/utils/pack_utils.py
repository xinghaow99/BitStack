import torch

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