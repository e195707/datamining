import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib 
import pandas as pd

# 演習1.1
def true_function(x):
    """
    >>> true_function(0) == 0
    True
    """
    return np.sin(np.pi * x * 0.8) * 10

def plot_truedata(x_min, x_max, dataset):
    dots = 100
    xs = np.linspace(x_min, x_max, dots)
    ys = true_function(xs)
    plt.figure()
    plt.plot(xs, ys, label="true_function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend(loc="best")
    #plt.show()
    plt.savefig("ex1.1.png")

    data_xs = dataset["観測点"].values
    data_ys = dataset["真値"].values
    plt.scatter(data_xs, data_ys, label="true_dataset")
    plt.legend(loc="best")
    #plt.show()
    plt.savefig("ex1.2.png")

    data_noisy_ys = dataset["観測値"].values
    plt.scatter(data_xs, data_noisy_ys, label="観測値")
    plt.legend(loc="best")
    #plt.show()
    plt.savefig("ex1.3.png")
    
# 演習1.2
def make_true_dataset(x_min, x_max, num_of_samples=20):
    np.random.seed(0)
    xs = np.random.rand(num_of_samples) * (x_max - x_min) + x_min
    xs.sort()
    ys = true_function(xs)
    temp = np.array([xs, ys]).T
    dataset = pd.DataFrame(temp, columns=["観測点", "真値"])
    return dataset

# 演習1.3
def make_dataset(dataset: pd.DataFrame, average=0.0, std=2.0) -> pd.DataFrame:
    size = len(dataset)
    noisy = np.random.normal(loc=average, scale=std, size=size)
    noisy_data = dataset["真値"] + noisy
    dataset["観測値"] = noisy_data
    return dataset

if __name__ == "__main__":
    x_min = -1
    x_max = 1
    dataset = make_true_dataset(x_min, x_max) # 真値生成
    dataset = make_dataset(dataset) # ノイズ付与した観測値生成
    plot_truedata(x_min, x_max, dataset)