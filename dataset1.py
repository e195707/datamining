import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib 

# 演習1.1
def true_function(x):
    """
    >>> true_function(0) == 0
    True
    """
    return np.sin(np.pi * x * 0.8) * 10

def plot_rawdata(x_min, x_max):
    dots = 100
    xs = np.linspace(x_min, x_max, dots)
    ys = true_function(xs)
    plt.figure()
    plt.plot(xs, ys, label="true_function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend(loc="best")
    plt.show()
    plt.savefig("ex1.1.png")



if __name__ == "__main__":
    x_min = -1
    x_max = 1
    plot_rawdata(x_min, x_max)
