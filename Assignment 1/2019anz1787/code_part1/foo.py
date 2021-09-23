"""
Demo to show importing function
"""
import numpy as np


def demo(args):
    print("foo imported successfully")
    print(args)
    k = 10
    weights = np.random.random(k+1)
    print(f"Polynomial={k}")
    print(f"weights={weights}")