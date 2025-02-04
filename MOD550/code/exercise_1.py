import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
import os
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
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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

                file_path = os.path.join(self.base_dir, "data", "my_data", "original_dataset.png")
                plt.savefig(file_path)
                #print("Plot saved as original_dataset.png")

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
        y = np.random.randint(-500, 501, size = (self.points, 1))

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


        file_path1 = os.path.join(self.base_dir, "data", "my_data", "dataset.csv")
        file_path2 = os.path.join(self.base_dir, "data", "my_data", "dataset.txt")

        np.savetxt(file_path1, self.data, delimiter=",", header="x,y", comments="")
        np.savetxt(file_path2, self.data, fmt="%.4f", delimiter="\t", header="x\ty")
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

                file_path = os.path.join(self.base_dir, "data", "my_data", "final_dataset.png")
                plt.savefig(file_path)
                print("Plot saved as final_dataset.png")

            if plot:
                plt.show()
        return self.data

    def save_metadata(self, filename="metadata.json"):
        """
        Saves metadata about dataset generation.
        """

        file_path = os.path.join(self.base_dir, "data", "my_data", "metadata.json")

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

        with open(file_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Metadata saved as {file_path}")

dataset = Dataset(100, 1, 2, 16, 100)
dataset.data_noise_around_function(plot = True)
dataset.random_data_uniform()
dataset.add_data(plot = True)
dataset.save_metadata("metadata.json")

# ________________________________________________________________________________________
# problems 6 - 8
import pandas as pd

# Data set was imported from https://github.com/SaiPieGiera/MOD550.git (Urszula Starowicz)
# read data
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
file_path1 = os.path.join(base_dir, "data", "imported_data", "imported_data_from_student.csv")
df_import = pd.read_csv(file_path1)

print(df_import.info())
print(df_import.describe())
# read metadata
file_path2 = os.path.join(base_dir, "data", "imported_data", "imported_metadata_from_student.json")
with open(file_path2 , "r", encoding="utf-8") as file:
    metadata_import = json.load(file)
print(metadata_import)

plt.figure(figsize=(8, 6))
plt.scatter(df_import["X"], df_import["Y"], alpha=0.5, color="blue", label="Imported Data")

plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Scatter Plot of Imported Data")
plt.legend()
plt.grid(True)
plt.show()

# Based on the metadata and scatter plot, the data points appear to be random
plt.hist(df_import["X"], bins=20, alpha=0.7, label="X values")
plt.hist(df_import["Y"], bins=20, alpha=0.7, label="Y values")
plt.legend()
plt.title("Histogram of X and Y of imported data")
plt.show()

correlation = df_import["X"].corr(df_import["Y"])
print(f"Correlation between X and Y of imported data: {correlation}")

# updating metadata

metadata_new_import = {
    "source": "Imported dataset from student Urszula Starowicz",
    "original_metadata": {
        "assignment": "Dataset Generation",
        "author": {
            "name": "Urszula Starowicz",
            "email": "283591@uis.no",
            "github": "SaiPieGiera"
        },
        "description": "This dataset consists of two randomly generated datasets combined together.",
        "parameters": {
            "num_points": 200,
            "noise_level": 16
        },
        "dataset_details": {
            "total_points": 400,
            "dataset_1_points": 200,
            "dataset_2_points": 200,
            "mean_values": [0.4975, 0.5166],
            "std_dev_values": [0.2891, 0.2959]
        }
    },
    "analysis": {
        "correlation": 0.0839,
        "distribution": "Both X and Y appear uniformly distributed with added noise.",
        "assumed_truth": "No meaningful relationship exists between X and Y. The dataset is random."
    },
    "conclusion": "This dataset is suitable for testing random data behavior but not for regression tasks."
}
file_path3 = os.path.join(base_dir, "data", "imported_data", "updated_imported_metadata_from_student.json")
with open(file_path3, "w") as f:
    json.dump(metadata_new_import, f, indent=4)

#_____________________________________________________________________________________________
# For regression task was used open 2D dataset from:
# https://github.com/hardikkamboj/An-Introduction-to-Statistical-Learning?tab=readme-ov-file

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# read data
file_path4 = os.path.join(base_dir, "data", "imported_data", "Auto.csv")
df_auto = pd.read_csv(file_path4)
print(df_auto.info())
#print(df_auto.describe())

# read metadata
file_path5 = os.path.join(base_dir, "data", "imported_data", "metadata_auto.json")
with open(file_path5, "r", encoding="utf-8") as file:
    metadata_import = json.load(file)
print(metadata_import)

# predicting fuel consumption (mpg) based on engine power (horsepower)
df = df_auto[['mpg', 'horsepower']].dropna()

# convert miles per gallon to km/L
df['km_per_liter'] = df['mpg'] * 0.425144

# if there is some non-numeric data in the 'horsepower' column we change it to NaN
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df = df.dropna()  # drop rows with NaN values

plt.figure(figsize=(8, 6))
plt.scatter(df['horsepower'], df['km_per_liter'], alpha=0.6, color='blue', label='Data Points')
plt.xlabel("Horsepower")
plt.ylabel("Fuel Efficiency (km/L)")
plt.title("Scatter Plot: Fuel Efficiency vs Engine Power")
plt.legend()
plt.grid(True)
plt.show()

# horsepower^2 feature
df['horsepower^2'] = df['horsepower'] ** 2

X = df[['horsepower', 'horsepower^2']]
y = df['km_per_liter']

# Fit the regression model
# We can treat now horsepower and horsepower^2 as independent features,
# i.e. still use linear regression but with 2 features:
# km_per_liter = beta0 + beta1 * horsepower + beta2 * horsepower^2
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

sorted_indices = np.argsort(df['horsepower'].values)
sorted_hp = df['horsepower'].values[sorted_indices]
sorted_y_pred = y_pred[sorted_indices]

mse = mean_squared_error(y, y_pred)

print(f"beta0: {model.intercept_}")
print(f"(beta1, beta2): {model.coef_}")
print(f"MSE: {mse}")

plt.scatter(df['horsepower'], y, color='blue', alpha=0.5, label="Actual km/L")
plt.plot(sorted_hp, sorted_y_pred, color='red', linewidth=2, label="Quadratic Fit")

plt.xlabel("Horsepower")
plt.ylabel("Kilometers per Liter (km/L)")
plt.title("Quadratic Regression: km/L vs Horsepower")
file_path6 = os.path.join(base_dir, "data", "imported_data", "guessed_truth_function_Autodata.png")
plt.savefig(file_path6)
plt.legend()
plt.grid()
plt.show()

# updating metadata
updated_metadata_auto = {
    "title": "Auto Dataset",
    "source": "An Introduction to Statistical Learning",
    "description": "This dataset contains information on various car models, including attributes such as horsepower, weight, acceleration, and fuel efficiency (mpg). It is commonly used for statistical learning and regression analysis.",
    "file_name": "Auto.csv",
    "file_type": "CSV",
    "size": "18 KB",
    "columns": [
        {
            "name": "mpg",
            "description": "Miles per gallon (fuel efficiency in the original dataset)"
        },
        {
            "name": "horsepower",
            "description": "Engine power (used in regression analysis)"
        },
        {
            "name": "km_per_liter",
            "description": "Fuel efficiency converted to kilometers per liter"
        }
    ],
    "source_usage": "Used for teaching statistical learning, regression, and data analysis techniques.",
    "license": "Educational use, refer to the book for specific licensing details.",
    "author": "Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani",
    "publication_year": "2013",
    "source_url": "https://www.statlearning.com/resources-second-edition",
    "analysis": {
        "objective": "Predicting fuel efficiency (km/L) based on engine power (horsepower).",
        "regression_type": "Quadratic Regression",
        "model_formula": "km_per_liter = beta0 + beta1 * horsepower + beta2 * horsepower^2",
        "coefficients": {
            "beta0": model.intercept_,
            "beta1": model.coef_[0],
            "beta2": model.coef_[1]
        },
        "mse": mse,
        "assumptions": [
            "Fuel efficiency decreases as engine power increases (negative beta1).",
            "The quadratic term (beta2) slightly compensates for the decrease at higher horsepower levels."
        ],
        "conclusion": "The dataset supports a quadratic relationship between fuel efficiency and engine power with a Mean Squared Error (MSE) of 3.43."
    }
}
file_path7 = os.path.join(base_dir, "data", "imported_data", "updated_metadata_auto.json")
with open(file_path7, "w", encoding="utf-8") as f:
    json.dump(updated_metadata_auto, f, indent=4)

#__________________________________________________________________________________

# problem 8: Select 3 github/gitlab repository and make an assessment on the coding standards they use.
# Github 1: https://github.com/daniilkrichevskiyDM/AI_for_engineers_UiS_Dementia.git
# Github 2: https://github.com/tensorflow/tensorflow
# Github 3: https://github.com/comp-physics/Quantum-PDE-Benchmark
Github1 = {
    '1. Readability and Clarity': (
        '\n- All files in the repository are clearly named and correspond to their purpose, ensuring easy navigation and understanding.'
        '\n- Code is supported by comments explaining key parts of the logic.'
        '\n- The repository does not include a metadata file, which could be useful for additional context.'
    ),
    '2. Structure': (
        '\n- Logical folder structure separates data, models, and scripts, making the repository easy to navigate.'
        '\n- Code is modular, utilizing functions and classes for better reusability.'
    ),
    '3. Consistency and Style': (
        '\n- Variable naming, function naming, and class naming follow consistent conventions.'
    ),
    '4. Documentation': (
        '\n- Not all functions and classes contain descriptions of their purpose and usage.'
        '\n- The repository includes a README file explaining the project\'s purpose and usage.'
    ),
    '5. Summary': (
        '\n- The repository follows good coding practices in readability, structure, and documentation. However, the absence of metadata files and comprehensive testing affects maintainability. Adding unit tests and improving documentation would enhance its robustness.'
    )
}

Github2 = {
    '1. Readability and Clarity': (
        '\n- All files in the repository are clearly named and correspond to their purpose, ensuring easy navigation and understanding.'
        '\n- Code is supported by comments explaining key parts of the logic.'
        '\n- The repository does not include a metadata file, which could be useful for additional context.'
    ),
    '2. Structure': (
        '\n- Logical folder structure separates core functionalities, tools, and third-party resources.'
        '\n- Code is modular, utilizing functions and classes for better reusability and maintainability.'
    ),
    '3. Consistency and Style': (
        '\n- Variable naming, function naming, and class naming follow consistent conventions.'
        '\n- The repository includes linting and formatting configurations (e.g., `.pylintrc`, `.clang-format`) to enforce coding standards.'
    ),
    '4. Documentation': (
        '\n- Includes additional documentation files like `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `SECURITY.md` to guide contributors and users.'
        '\n- Contains a comprehensive `README.md` file explaining the project, its purpose, installation, and usage.'
    ),
    '5. Summary': (
        '\n- The TensorFlow repository exemplifies high standards in coding practices, with well-structured, readable, and well-documented code. Its commitment to testing, error handling, and maintainability ensures its reliability and usability'
    )
}

Github3 = {
    '1. Readability and Clarity': (
        '\n- All files in the repository are clearly named and correspond to their purpose, ensuring easy navigation and understanding.'
        '\n- The code includes comments explaining key parts of the logic, enhancing comprehensibility.'
    ),
    '2. Structure': (
        '\n- The repository exhibits a logical folder structure, separating core functionalities, data, and tutorials.'
        '\n- Code is modular, utilizing functions and classes for better reusability and maintainability.'
    ),
    '3. Consistency and Style': (
        '\n- Variable, function, and class naming conventions are consistent throughout the codebase.'
        '\n- The repository includes a .gitignore file to manage unnecessary files.'
    ),
    '4. Documentation': (
        '\n- Contains a comprehensive README.md file explaining the project\'s purpose, installation, and usage.'
        '\n- Includes a LICENSE file.'
    ),
    '5. Summary': (
        '\n- The Quantum-PDE-Benchmark repository demonstrates good practices in code readability, structure, and documentation. Implementing unit tests would significantly improve its robustness and usability.'
    )
}

try:
    file_path8 = os.path.join(base_dir, "code", "coding_standards_3_github.txt")
    with open(file_path8, 'w') as f:
        f.write('MOD550 task 8\n')

        f.write('\nGithub 1: https://github.com/daniilkrichevskiyDM/AI_for_engineers_UiS_Dementia.git\n')
        for key, value in Github1.items():
            f.write(f'{key}: {value}\n')

        f.write('\nGithub 2: https://github.com/tensorflow/tensorflow\n')
        for key, value in Github2.items():
            f.write(f'{key}: {value}\n')

        f.write('\nGithub 3: https://github.com/comp-physics/Quantum-PDE-Benchmark\n')
        for key, value in Github3.items():
            f.write(f'{key}: {value}\n')

    print("Text file saved.")
except Exception as e:
    print(f"Error while saving the file: {e}")