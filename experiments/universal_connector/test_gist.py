import torch
import torch.nn as nn
from gist_memory.core import GistLayer

torch.manual_seed(42)
B, L, D = 4, 128, 64

# Dummy training loop to learn identity mapping from x to y across sequence
def test_kernel(kernel_name):
    layer = GistLayer(d_model=D, d_map=16, decay=0.99, kernel=kernel_name, use_salience_gate=False)
    opt = torch.optim.Adam(layer.parameters(), lr=0.01)
    
    # We want the layer to reconstruct the input vector shifted by some lag or just reconstruct random targets
    target = torch.randn(B, L, D)
    
    for step in range(50):
        x = torch.randn(B, L, D)
        out, _ = layer(x)
        loss = nn.functional.mse_loss(out, target)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    return loss.item()

print(f"Final Loss (Square Kernel): {test_kernel('square'):.4f}")
print(f"Final Loss (Prime Kernel):  {test_kernel('prime'):.4f}")
