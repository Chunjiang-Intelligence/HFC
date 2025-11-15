# hfc/training/compressor.py
import torch

class GradientCompressor:
    def __init__(self, top_k_ratio=0.01, quant_bits=8):
        self.top_k_ratio = top_k_ratio
        self.quant_bits = quant_bits
        self.quant_max = 2**(self.quant_bits - 1) - 1
        self.quant_min = -2**(self.quant_bits - 1)

    def compress(self, tensor: torch.Tensor):
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()
        
        k = max(1, int(len(tensor_flat) * self.top_k_ratio))
        values, indices = torch.topk(tensor_flat.abs(), k, largest=True, sorted=False)
        top_k_values = tensor_flat[indices]

        if k == 0: return None

        scale = top_k_values.abs().max() / self.quant_max
        if scale == 0: scale = 1.0
            
        quantized_values = torch.clamp(
            torch.round(top_k_values / scale), self.quant_min, self.quant_max
        ).to(torch.int8)

        return {
            "quantized_values": quantized_values,
            "indices": indices,
            "original_shape": original_shape,
            "scale": scale,
        }

    def decompress(self, compressed_data: dict, device):
        if compressed_data is None: return None

        dequantized_values = compressed_data["quantized_values"].to(device).float() * compressed_data["scale"]
        indices = compressed_data["indices"].to(device)
        original_shape = compressed_data["original_shape"]
        num_elements = torch.prod(torch.tensor(original_shape))
        
        decompressed_flat = torch.zeros(num_elements, dtype=torch.float32, device=device)
        decompressed_flat.scatter_(0, indices, dequantized_values)

        return decompressed_flat.view(original_shape)
