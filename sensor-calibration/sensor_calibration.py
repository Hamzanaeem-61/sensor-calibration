# Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Generate Synthetic "Uncalibrated" Lab Data
torch.manual_seed(42)

# Simulate raw resistance readings from 10 ohms to 500 ohms
raw_resistance = torch.linspace(10, 500, steps=100)

# Calculate the "true" temperature based on physics, then add sensor noise
true_a, true_b, true_c = 500.0, 2.5, 50.0
true_temperature = (true_a / (torch.log(raw_resistance) + true_b)) + true_c

# Add random Gaussian noise to simulate realistic analog sensor jitter
noise = torch.randn(raw_resistance.size()) * 1.5
noisy_temperature = true_temperature + noise

# Reshape data into column vectors to match standard model inputs (like the StreetEasy dataset)
X = raw_resistance.view(-1, 1)
y = noisy_temperature.view(-1, 1)

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    train_size=0.7, 
    test_size=0.3, 
    random_state=2
)

# 2. Create a Custom Calibration Model
class CalibrationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize learnable parameters as native nn.Parameters
        self.a = nn.Parameter(torch.tensor(100.0))
        self.b = nn.Parameter(torch.tensor(1.0))
        self.c = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        # Forward Pass: Predict Temperature using our calibration equation
        return (self.a / (torch.log(x) + self.b)) + self.c

model = CalibrationModel()

# 3. MSE loss function + optimizer
loss_fn = nn.MSELoss()
# Using Adam optimizer (with an adjusted learning rate to scale up to the true parameter values)
optimizer = optim.Adam(model.parameters(), lr=0.5)

# 4. Training Loop
num_epochs = 5000
print("--- Starting Calibration ---")
for epoch in range(num_epochs):
    predictions = model(X_train) 
    MSE = loss_fn(predictions, y_train) 
    
    # Backward pass and optimization step
    MSE.backward()
    optimizer.step() 
    optimizer.zero_grad()
    
    # Keep track of the loss and parameters during training
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}] | MSE Loss: {MSE.item():.4f} | '
              f'a: {model.a.item():.2f}, b: {model.b.item():.2f}, c: {model.c.item():.2f}')

print("\n--- Calibration Complete ---")

import os

# Save model
os.makedirs('models', exist_ok=True)
MODEL_PATH = os.path.join('models', 'calibration_model.pth')
torch.save(model, MODEL_PATH)

# Load model
torch.serialization.add_safe_globals([CalibrationModel])
loaded_model = torch.load(MODEL_PATH)
loaded_model.eval()

with torch.no_grad():
    predictions = loaded_model(X_test)
    test_MSE = loss_fn(predictions, y_test)

# Show output
print('Test MSE is ' + str(test_MSE.item()))
print('Test Root MSE is ' + str(test_MSE.item()**(1/2)))
print(f"\nTarget Values -> a: {true_a}, b: {true_b}, c: {true_c}")
print(f"Fitted Values -> a: {loaded_model.a.item():.2f}, b: {loaded_model.b.item():.2f}, c: {loaded_model.c.item():.2f}")

# 5. Plotting Results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, label='Predictions', alpha=0.6, color='blue')

plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values')

# Generate ideal diagonal line coordinates
min_val = min(y_test).item()
max_val = max(y_test).item()
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray', linewidth=2, label="Perfect Calibration")

plt.legend()
plt.title('Sensor Calibration - Predictions vs Actual Values')

os.makedirs('plots', exist_ok=True)
plt.savefig(os.path.join('plots', 'calibration_results.png'), dpi=150, bbox_inches='tight')
plt.show()