import os
import subprocess
from itertools import product

batch_sizes = [8, 16, 32, 64, 128]
learning_rates = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
BASE_OUT = "./search"
epochs = 15

def lr_str(lr: float) -> str:
    s = f"{lr:.6f}".rstrip("0").rstrip(".")
    return s

def run_one(bs: int, lr: float):
    out_dir = os.path.join(BASE_OUT, f"bs_{bs}_lr_{lr_str(lr)}")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
            "python", "train.py",
            "-bs", str(bs),
            "-lr", str(lr),
            "-e", str(epochs),
            "-o", out_dir,
        ]
    
    subprocess.run(cmd, check=True)

os.makedirs(BASE_OUT, exist_ok=True)
combos = list(product(batch_sizes, learning_rates))
total = len(combos)

for i, (bs, lr) in enumerate(combos, 1):
    print(f"Running: bs={bs}, lr={lr}")
    run_one(bs, lr)
