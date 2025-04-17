import tkinter as tk
from PIL import Image, ImageDraw, ImageTk, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define same neural network architecture as in mnist.py
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Load the trained model
model = SimpleNN()
trained_model_path = "MNIST\mnist_model_best.pth"

try:
    model.load_state_dict(torch.load(trained_model_path, map_location="cpu"))
    model.to(device)
    model.eval()
    print(f"Model loaded successfully to {device}.")
except FileNotFoundError:
    print("Model file not found. Please train the model first.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# MNIST normalization values (same as training)
mean = 0.1307
std = 0.3081

# Define transformation for handdrawn digit
preprocess_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

# GUI setup using Tkinter
window = tk.Tk()
window.title("MNIST Digit Predictor")

canvas_width = 280
canvas_height = 280

# Create the main canvas to draw on
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="white", cursor="pencil")
canvas.pack(pady=10)

# Create a PIL image for drawing in parallel
pil_image = Image.new("L", (canvas_width, canvas_height), "white")
pil_draw = ImageDraw.Draw(pil_image)

result_label = tk.Label(window, text="Draw a digit", font=("Helvetica", 14))
result_label.pack(pady=5)

# Drawing functionality
drawing = False
last_x, last_y = None, None
line_width = 15

def start_drawing(event):
    global drawing, last_x, last_y
    drawing = True
    last_x, last_y = event.x, event.y

def draw(event):
    global last_x, last_y
    if drawing:
        x, y = event.x, event.y
        canvas.create_line(last_x, last_y, x, y, fill="black", width=line_width, joinstyle="round")
        pil_draw.line([last_x, last_y, x, y], fill="black", width=line_width, joint="round")
        last_x, last_y = x, y

def stop_drawing(event):
    global drawing
    drawing = False

canvas.bind("<Button-1>", start_drawing)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_drawing)

# Drawing preprocessing
def preprocess(image):
    img_inverted = ImageOps.invert(image)
    img_resized = img_inverted.resize((28, 28), Image.LANCZOS)
    img_tensor = preprocess_transform(img_resized).unsqueeze(0)
    return img_tensor

# Predict drawing
def predict_digit():
    if not canvas.find_all():
        result_label.config(text="Please draw a digit!")
        return
    
    img_tensor = preprocess(pil_image)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)
        output_cpu = output.cpu()
        probabilities = F.softmax(output_cpu, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_digit = predicted_idx.item()
        confidence_percentage = confidence.item() * 100

    result_label.config(text=f"Predicted: {predicted_digit} ({confidence_percentage:.2f}%)")

# Clear canvas function
def clear_canvas():
    global pil_image, pil_draw
    canvas.delete("all")
    pil_image = Image.new("L", (canvas_width, canvas_height), "white")
    pil_draw = ImageDraw.Draw(pil_image)
    result_label.config(text="Draw a digit")

# Buttons
button_frame = tk.Frame(window)
button_frame.pack(pady=10)

clear_button = tk.Button(button_frame, text="Clear", command=clear_canvas, font=("Helvetica", 12), width=10)
clear_button.pack(side=tk.LEFT, padx=5)

predict_button = tk.Button(button_frame, text="Predict", command=predict_digit, font=("Helvetica", 12), width=10)
predict_button.pack(side=tk.LEFT, padx=5)

# Start main loop
window.mainloop()