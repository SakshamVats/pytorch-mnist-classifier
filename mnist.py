import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

# Transformations for MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load MNIST training
train_dataset = datasets.MNIST(
    root="MNIST/data",
    train=True,
    download=True,
    transform=transform
)

# Download and load MNIST test
test_dataset = datasets.MNIST(
    root="MNIST/data",
    train=False,
    download=True,
    transform=transform
)

# Create DataLoaders (to handle batching and shuffling)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Define the Neural Network
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
    
# Initialize model, loss function, and optimizer
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
def train(epoch):
    model.train() # training mode
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % 100 == 99:
            print(f"Epoch: {epoch+1} | Batch: {batch_idx+1:5d} | Loss: {running_loss/100:.4f}")
            running_loss = 0.0

# Evaluation loop
def evaluate():
    model.eval()
    test_loss = 0.0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += criterion(outputs, target).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f"Test Loss: {test_loss:.4f} | Accuracy: {accuracy:.2f}%")
    return accuracy

# Run training and then evaluate
if __name__ == "__main__":
    start = time.time()

    best_accuracy = 0.0
    for epoch in range(EPOCHS):
        print(F"--- Starting Epoch {epoch+1} ---")
        train(epoch)
        current_accuracy = evaluate()

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            torch.save(model.state_dict(), "MNIST/mnist_model_best.pth")
            print(f"Saved new best model with accuracy: {best_accuracy:.2f}%")

    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds")