import sys

import cleanlab
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision


def main() -> None:
    print("=" * 50)
    print("Environment information")
    print("=" * 50)

    print(f"Python:       {sys.version}")
    print(f"Python path:  {sys.executable}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Torchvision:  {torchvision.__version__}")
    print(f"Cleanlab:     {cleanlab.__version__}")
    print(f"NumPy:        {np.__version__}")
    print(f"Pandas:       {pd.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")

    print("-" * 50)
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA build: {torch.version.cuda}")
    else:
        print("GPU is not currently available. Training will use CPU.")

    print("=" * 50)


if __name__ == "__main__":
    main()