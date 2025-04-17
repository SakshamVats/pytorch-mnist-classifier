# Handwritten Digit Recognition using PyTorch and Tkinter

## Overview

This project demonstrates a simple handwritten digit recognition system using the MNIST dataset. It consists of two main parts:

1.  **`mnist.py`**: A Python script that trains a neural network model on the MNIST dataset using the PyTorch library.
2.  **`draw_predict.py`**: A Python script that uses the trained model to predict digits drawn by the user on a graphical interface created with Tkinter.

## Key Features

* **MNIST Dataset Training:** Trains a simple feedforward neural network on the MNIST dataset of handwritten digits.
* **Real-time Drawing:** Allows users to draw digits on a canvas using their mouse.
* **Digit Prediction:** Uses the trained PyTorch model to predict the digit drawn by the user.
* **Confidence Score:** Displays the model's confidence level for its prediction.
* **Processed Image Visualization:** Shows the image after preprocessing, which is what the model actually receives as input.

## Prerequisites

Before running the project, make sure you have the following installed:

* **Python 3.x**
* **PyTorch** (`torch` and `torchvision`)
* **Pillow (PIL)** (`PIL`)
* **Tkinter** (usually included with standard Python installations)

You can install the necessary PyTorch and Pillow libraries using pip:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install Pillow
```

**Note:** If you have a CUDA-enabled GPU and want to use it for training, you might need to install a different version of PyTorch. Refer to the official PyTorch website for installation instructions.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd [YOUR_REPOSITORY_NAME]
    ```

2.  **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt  # If you create a requirements.txt file
    # Or install them directly as mentioned in the Prerequisites section
    ```

## Usage

### 1. Training the Model (Optional)

If you want to train the model yourself (a pre-trained model is included), run the `mnist.py` script:

```bash
python MNIST/mnist.py
```

This script will download the MNIST dataset, train a simple neural network, and save the best model weights to `MNIST/mnist_model_best.pth`.

### 2. Running the Digit Predictor

To run the handwritten digit prediction GUI, execute the `draw_predict.py` script:

```bash
python draw_predict.py
```

This will open a window where you can draw a digit using your mouse. Click the "Predict" button to see the model's prediction and confidence score. You can also click "Show Processed" to see the image that the model is actually receiving after preprocessing.

## Project Structure

```
your_project_name/
├── MNIST/
│   ├── data/               # Folder where MNIST dataset is downloaded
│   ├── mnist.py            # Script to train the neural network
│   └── mnist_model_best.pth  # Pre-trained model weights (will be created after training)
├── draw_predict.py       # Script for the drawing and prediction GUI
├── README.md             # This file
└── requirements.txt        # (Optional) List of project dependencies
```

You can create a `requirements.txt` file by running:

```bash
pip freeze > requirements.txt
```

## Model

The project uses a simple feedforward neural network with three fully connected layers. The architecture is defined in both `mnist.py` and `draw_predict.py`. The training script saves the model with the best validation accuracy to `MNIST/mnist_model_best.pth`.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

[Specify your project license here, e.g., MIT License](https://opensource.org/licenses/MIT)

## Acknowledgments

* The MNIST dataset is a widely used dataset for handwritten digit recognition.
* This project utilizes the PyTorch library for building and training the neural network.
* The graphical user interface is built using the Tkinter library.
