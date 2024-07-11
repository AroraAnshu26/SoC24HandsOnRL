import json
import numpy as np
import matplotlib.pyplot as plt

def inv_transform(distribution: str, num_samples: int, **kwargs) -> list:
    """ populate the 'samples' list from the desired distribution """

    samples = []

    # TODO: first generate random numbers from the uniform distribution

    uniform_samples = np.random.uniform(0, 1, num_samples)   #generating the uniform samples first
    
    if distribution == "exponential":
        lambda_param = kwargs.get('lambda')                  #from kwargs, getting the given lambda
        samples = [round(-np.log(1 - u) / lambda_param, 4) for u in uniform_samples]   # Apply the inverse CDF transformation (we know what it is for exponential) and cappind the decimal places to 4

        
    elif distribution == "cauchy":
        # Apply the inverse CDF transformation for standard Cauchy distribution, and roudnign off to 4 decimal places
        gamma_param = kwargs.get('gamma')
        x_peak_param = kwargs.get('peak_x')
        samples = [round((np.tan(np.pi * (u - 0.5))*gamma_param+ x_peak_param), 4) for u in uniform_samples]

    # END TODO
            
    return samples


if __name__ == "__main__":
    np.random.seed(42)

    for distribution in ["cauchy", "exponential"]:
        file_name = "q1_" + distribution + ".json"
        args = json.load(open(file_name, "r"))
        samples = inv_transform(**args)
        
        with open("q1_output_" + distribution + ".json", "w") as file:
            json.dump(samples, file)

        # TODO: plot and save the histogram to "q1_" + distribution + ".png"
        
        
        plt.figure()
        plt.hist(samples, bins=50, density=True, alpha=0.6, color='g')
        plt.title(f"Histogram samples from {distribution} distribution")
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f"q1_{distribution}.png")
        plt.close()
        # END TODO
       
       
    