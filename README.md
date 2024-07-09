# Perceptron Implementation

This repository contains a simple implementation of a Perceptron algorithm in Python using NumPy. The Perceptron is a basic neural network unit used for binary classification tasks.

## Project Description

The project demonstrates the following:

- Initialization of a Perceptron model.
- Training the model on a simple AND gate dataset.
- Evaluating the model's performance.
- Predicting outputs using the trained model.

## Code Explanation

### Perceptron Class

The `Perceptron` class has the following methods:

- `__init__(self, num_features, learning_rate=0.1, epochs=100)`: Initializes the perceptron with weights, learning rate, and number of epochs.
- `activation(self, z)`: Activation function that returns 1 if input `z` is greater than or equal to 0, else returns 0.
- `predict(self, x)`: Predicts the output for a given input `x`.
- `train(self, X, y)`: Trains the perceptron using the input data `X` and labels `y`.
- `evaluate(self, X, y)`: Evaluates the perceptron on test data and returns the accuracy.

### Usage

1. **Training Data**: The dataset used here is for an AND gate:
    ```python
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    ```

2. **Initialize and Train the Perceptron**:
    ```python
    perceptron = Perceptron(num_features=2)
    perceptron.train(X, y)
    ```

3. **Evaluate the Model**:
    ```python
    accuracy = perceptron.evaluate(X, y)
    print(f'Accuracy: {accuracy * 100}%')
    ```

4. **Test the Perceptron**:
    ```python
    for x in X:
        print(f'Input: {x}, Predicted Output: {perceptron.predict(x)}')
    ```

## Getting Started

### Prerequisites

- Python 3.x
- NumPy

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/perceptron.git
    ```
2. Navigate to the project directory:
    ```sh
    cd perceptron
    ```

### Running the Code

1. Ensure you have the required libraries installed:
    ```sh
    pip install numpy
    ```
2. Run the script:
    ```sh
    python perceptron.py
    ```

## Acknowledgments

- The Perceptron algorithm is a fundamental machine learning algorithm for binary classification tasks.

Feel free to explore, provide feedback, and contribute to this project!
