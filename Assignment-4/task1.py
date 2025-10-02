import numpy as np
import matplotlib.pyplot as plt

sample_sizes = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for n in sample_sizes:
    # Step 1: Simulate two dice throws n times
    dice1 = np.random.randint(1, 7, n)
    dice2 = np.random.randint(1, 7, n)
    sums = dice1 + dice2

    values, bins = np.histogram(sums, bins=np.arange(2, 14), density=True)

    plt.figure(figsize=(8, 5))
    plt.bar(bins[:-1], values, width=0.9, align="center", alpha=0.7, edgecolor="black")
    plt.xlabel("Sum of Two Dice")
    plt.ylabel("Relative Frequency")
    plt.title(f"Histogram of Dice Sums (n={n})")
    plt.xticks(range(2, 13))
    plt.show()

    print(f"\nResults for n={n}:")
    print("Frequencies:", np.round(values, 3))
