import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
with open("output.txt", "w") as f:
    f.write(f"PyTorch version: {torch.__version__}\n")
    f.write(f"CUDA available: {torch.cuda.is_available()}\n")