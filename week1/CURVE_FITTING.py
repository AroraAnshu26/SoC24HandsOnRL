from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def func(t, v, k):
    """Computes the function S(t) with constants v and k."""
    return v * (t - (1 - np.exp(-k * t))/k)

def find_constants(df: pd.DataFrame, func: Callable):
    """Returns the constants v and k."""
    # Extract t and S(t) from the DataFrame
    t_data = df['t']
    s_data = df['S']

    # Fit the curve using curve_fit
    popt, pcov = curve_fit(func, t_data, s_data)

    # Extract the optimized parameters
    v, k = popt

    return v, k

if __name__ == "__main__":
    # Read the data
    df = pd.read_csv(r"C:\Users\anshu\Desktop\HANDS_ON_RL_SoC\week1\week1\q3\data.csv")

    # Estimate v and k
    v, k = find_constants(df, func)
    v = round(v, 4)
    k = round(k, 4)
    print("Estimated Constants:")
    print("v:", v)
    print("k:", k)

    # Plot the experimental data and the fitted curve
    plt.figure(figsize=(8, 6))
    plt.scatter(df['t'], df['S'], label='Experimental Data')
    t_values = np.linspace(df['t'].min(), df['t'].max(), 100)
    plt.plot(t_values, func(t_values, v, k), 'r-', label='Fitted Curve')
    plt.xlabel('Time (t)')
    plt.ylabel('S(t)')
    plt.title('Experimental Data and Fitted Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('fit_curve.png')
    plt.show()
