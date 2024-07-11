
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PCA(init_array: pd.DataFrame, dimensions: int = 2):
    
    standardized_data = init_array - init_array.mean()    #follwoing all steps as given in pdf link
    covariance_matrix = np.cov(standardized_data.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = round(eigenvalues[sorted_indices],4)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    eigenvectors_subset = sorted_eigenvectors[:, :dimensions]
    final_data = np.dot(standardized_data, eigenvectors_subset)

    return sorted_eigenvalues, final_data

if __name__ == '__main__':
    # Read the data
    init_array = pd.read_csv("pca_data.csv", header=None)
    
    sorted_eigenvalues, final_data = PCA(init_array)

    np.savetxt("transform.csv", final_data, delimiter=',')
    
    for eig in sorted_eigenvalues:
        print(eig)

    # Plot and save a scatter plot of final_data to out.png
    plt.figure(figsize=(8, 8))
    plt.scatter(final_data[:, 0], final_data[:, 1])
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Result')
    plt.grid(True)
    plt.savefig('out.png')
    plt.show()

