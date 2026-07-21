import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    
    csv = pd.read_csv("csv/gratings/triangles.csv", sep = ",")
    # csv = csv * 10e6
    x = csv["x"]
    y = abs(-csv["y"])

    
    plt.figure(figsize=(8,5))
    plt.plot(x, y, color = "blue")
    
    # plt.plot(wavelength, A, label = "Absorption (R)", color = "red")
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.title("Gainazalaren geometria")
    plt.xlim(0,2*np.pi)
    plt.ylim(0,1)
    # plt.grid(True)
    plt.tight_layout()
    plt.show()