import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
 # Assignment_1 MOD550

# problems 1 - 5
class Dataset:
    """
    A class to generate and manipulate 2D datasets, including:
    - A quadratic function with Gaussian noise
    - A uniformly distributed random dataset

    Attributes:
        points (int): Number of data points.
        coeff1 (float): Coefficient for the quadratic term.
        coeff2 (float): Constant term in the quadratic function.
        mean (float): Mean of the Gaussian noise.
        sd (float): Standard deviation of the Gaussian noise.
    """

    def __init__(self,  points, coeff1, coeff2, mean, sd):
        """
        Initializes the dataset generator.
        """
        self.points = points
        self.coeff1 = coeff1
        self.coeff2 = coeff2
        self.mean = mean
        self.sd = sd
        self.data = None
        self.random_data = None
        self.initial_shape = None
        self.final_shape = None

    def data_noise_around_function(self, plot = False, save_plot = True):
        """
        Generates a 2D dataset with noise around a quadratic function.

        Args:
            plot (bool, optional): If True, displays the dataset plot.
            save_plot (bool, optional): If True, saves the plot to a file.

        Returns:
            np.ndarray: A 2D array containing x-values and noisy y-values.
        """
        x = np.random.randint(-20, 21, size = (self.points, 1))
        y = self.coeff1 * x**2 + self.coeff2
        noise = np.random.normal(self.mean, self.sd, size = x.shape)
        y_noise = y + noise

        self.data = np.hstack((x, y_noise))
        self.initial_shape = self.data.shape
        #print(f"Initial dataset shape: {self.initial_shape}")

        if plot or save_plot:
            # Commented code for plotting quadratic function
            #sorted_indices = np.argsort(x[:,0])
            #x_sorted = x[sorted_indices]
            #y_sorted = y[sorted_indices]

            plt.figure()
            plt.scatter(x, y_noise, color = "blue")
            # For this task it is not necessary to plot function
            #plt.plot(x_sorted, y_sorted, color = "black", label = f"Function: y = {self.coeff1} * x^2 + {self.coeff2}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Original dataset")
            plt.grid()

            if save_plot:
                plt.savefig("original_dataset.png")
                print("Plot saved as original_dataset.png")

            if plot:
                plt.show()

        return self.data

    def random_data_uniform(self):
        """
        Generates a random 2D dataset with uniform distribution.

        Returns:
            np.ndarray: A 2D array containing random x and y values.
        """
        x = np.random.randint(-20, 21, size = (self.points, 1))
        y = np.random.randint(-500, 500, size = (self.points, 1))

        self.random_data = np.hstack((x, y))
        #print(f"Random dataset shape:{self.random_data.shape}")
        return self.random_data

    def add_data(self, plot = False, save_plot = True):
        """
        Appends new data set, increasing number of examples (additional rows).
        Saves dataset as csv and txt file.

        Args:
            plot (bool, optional): If True, plots the generated dataset.
            save_plot (bool, optional): If True, saves the plot to a file.
        """
        if self.random_data is None:
            print("Error: `random_data_uniform()` must be called before `add_data()`")
            return

        if self.data is None:
            self.data = self.random_data
        else:
            self.data = np.append(self.data, self.random_data, axis=0)

        self.final_shape = self.data.shape
        #print(f"New dataset shape: {self.final_shape}")

        np.savetxt("dataset.csv", self.data, delimiter=",", header="x,y", comments="")
        np.savetxt("dataset.txt", self.data, fmt="%.4f", delimiter="\t", header="x\ty")
        print("Dataset saved as dataset.csv and dataset.txt")

        if plot or save_plot:
            num_samples = self.data.shape[0]
            # print("Number of samples:", num_samples)
            half_index = num_samples // 2
            # print("Half index:", half_index)
            original_data = self.data[:half_index, :]
            new_data = self.data[half_index:, :]

            plt.figure()
            plt.scatter(new_data[:, 0], new_data[:, -1], color = "green", label = "Original data")
            plt.scatter(original_data[:, 0], original_data[:, -1], color = "blue", label = "Added data")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.title("Final dataset")
            plt.grid()

            if save_plot:
                plt.savefig("final_dataset.png")
                print("Plot saved as final_dataset.png")

            if plot:
                plt.show()
        return self.data

    def save_metadata(self, filename="metadata.json"):
        """
        Saves metadata about dataset generation.
        """
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = {
            "Project": "2D Dataset Generator",
            "Author": "Svetlana Andrusenko",
            "Email": "283721@uis.no",
            "Created On": now,
            "Description": "This project generates and visualizes synthetic datasets using a quadratic function with noise and random uniform distribution.",
            "Data Details": {
                "Quadratic Data": {
                    "Points": self.initial_shape[0] if self.initial_shape else "Unknown",
                    "Equation": f"y = {self.coeff1} * x^2 + {self.coeff2} + noise",
                    "Noise Mean": self.mean,
                    "Noise Std": self.sd
                },
                "Random Data": {
                    "Points": self.points,
                    "X Range": "[-20, 20]",
                    "Y Range": "[-500, 500]"
                },
                "Final Dataset": {
                    "Initial Shape": self.initial_shape,
                    "Final Shape": self.final_shape,
                    "Output Files": ["dataset.csv", "dataset.txt"]
                }
            },
            "Execution": {
                "Python Version": "3.9+",
                "Dependencies": ["numpy", "matplotlib"],
                "Usage": "Run exercise_1.py to generate and visualize datasets."
            },
            "License": "MIT",
            "Ethical Considerations": "Synthetic data for educational use."
        }

        with open(filename, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved as {filename}")


dataset = Dataset(100, 1, 2, 16, 100)
dataset.data_noise_around_function(plot = True)
dataset.random_data_uniform()
dataset.add_data(plot = True)
dataset.save_metadata("metadata.json")


