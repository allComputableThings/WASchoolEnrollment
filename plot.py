import matplotlib.pyplot as plt
import numpy as np

def do_plot():
    # 1. Prepare your data
    x = np.linspace(0, 10, 100) # 100 evenly spaced points from 0 to 10
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = x**2

    # 2. Create the plot(s)

    # Plot 1: Sine wave
    plt.figure(figsize=(8, 6)) # Create a new figure with a specific size
    plt.plot(x, y1, label='Sine Wave', color='blue', linestyle='-')
    plt.title('Sine Wave Example')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.legend()

    # Plot 2: Cosine wave and a quadratic function on the same figure
    plt.figure(figsize=(10, 7)) # Another new figure
    plt.plot(x, y2, label='Cosine Wave', color='red', linestyle='--')
    plt.plot(x, y3, label='X Squared', color='green', linestyle=':')
    plt.title('Cosine and Quadratic Functions')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.legend()

    # You can also create subplots within a single figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 8)) # 2 rows, 1 column of subplots

    # First subplot
    axes[0].plot(x, y1, color='purple')
    axes[0].set_title('Sine Wave (Subplot 1)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Sin(X)')
    axes[0].grid(True)

    # Second subplot
    axes[1].scatter(x, y2, color='orange', s=10) # Using scatter plot for variety
    axes[1].set_title('Cosine Wave (Subplot 2)')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Cos(X)')
    axes[1].grid(True)

    plt.tight_layout() # Adjust subplot params for a tight layout

    # 3. Show the plot(s) in a window
    # This command will open all created figures in separate windows.
    # The program will pause here until all plot windows are closed.
    plt.show()

    print("All plots displayed and closed.")

if __name__ == "__main__":
    do_plot()