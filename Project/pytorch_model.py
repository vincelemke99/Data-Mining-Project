import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# -------------------------------------
# 1. PyTorch Model Definition
# -------------------------------------
class PyTorchLinearRegression(nn.Module):
    """Basic PyTorch Linear Regression Model."""
    def __init__(self, n_features):
        super().__init__()
        # Use standard nn.Linear since we know n_features during fit
        self.linear = nn.Linear(n_features, 1)
        # Optional: Initialize weights
        # self.linear.weight.data.normal_(0, 0.01)
        # self.linear.bias.data.fill_(0)

    def forward(self, X):
        return self.linear(X)

# -------------------------------------
# 2. Scikit-learn Wrapper Class
# -------------------------------------
class PyTorchRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    A scikit-learn wrapper for a PyTorch regression model.
    Handles data conversion, training loop, and prediction interface.
    """
    def __init__(self, lr=0.01, epochs=10, batch_size=32, device='cpu'):
        """
        Initialize the wrapper.
        Args:
            lr (float): Learning rate for the optimizer.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training DataLoader.
            device (str): 'auto', 'cuda', or 'cpu'. 'auto' uses GPU if available.
        """
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        # Internal attributes - initialized in fit
        self.model_ = None
        self.n_features_in_ = 0

    def _get_device(self):
        """Determine the computing device."""
        if self.device == 'auto':
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def fit(self, X, y):
        """
        Fits the internal PyTorch model to the training data.
        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features).
            y (np.ndarray): Training target, shape (n_samples,).
        Returns:
            self: The fitted estimator.
        """
        # 1. Validate input using sklearn utilities
        # Ensure float32 for PyTorch and require dense arrays
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float32, y_numeric=True)

        # Reshape y to be a 2D tensor [n_samples, 1] as expected by MSELoss
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # 2. Store input features and initialize the PyTorch model
        self.n_features_in_ = X.shape[1]
        # Instantiate the actual PyTorch model (defined above)
        self.model_ = PyTorchLinearRegression(n_features=self.n_features_in_)
        current_device = self._get_device()
        self.model_.to(current_device)
        print(f"PyTorch Wrapper: Using device {current_device}") # Added print statement

        # 3. Convert data to PyTorch Tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(current_device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(current_device)

        # 4. Create DataLoader for batching
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 5. Define Loss and Optimizer
        criterion = nn.MSELoss()
        # Using Adam is often a good starting point, but SGD is fine too
        # optimizer = optim.SGD(self.model_.parameters(), lr=self.lr)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        # 6. Training Loop
        self.model_.train() # Set model to training mode
        print(f"PyTorch Wrapper: Starting training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                # Standard PyTorch training steps
                optimizer.zero_grad()       # Zero gradients
                outputs = self.model_(batch_X) # Forward pass
                loss = criterion(outputs, batch_y) # Calculate loss
                loss.backward()             # Backward pass
                optimizer.step()            # Update weights
                epoch_loss += loss.item()   # Accumulate loss

            # Optional: Print progress (Corrected indentation)
            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                print(f'  Epoch [{epoch+1}/{self.epochs}], Avg Loss: {epoch_loss/len(loader):.4f}')

        # Mark as fitted according to scikit-learn convention
        self.is_fitted_ = True
        print("PyTorch Wrapper: Training finished.")
        return self

    # --- THESE METHODS MUST BE INSIDE THE CLASS ---

    def predict(self, X):
        """
        Makes predictions using the fitted PyTorch model.
        Args:
            X (np.ndarray): Features to predict on, shape (n_samples, n_features).
        Returns:
            np.ndarray: Predicted values, shape (n_samples,).
        """
        # 1. Check if the model has been fitted
        check_is_fitted(self, 'is_fitted_')

        # 2. Validate input
        X = check_array(X, accept_sparse=False, dtype=np.float32)

        # Check if the number of features matches the training data
        if X.shape[1] != self.n_features_in_:
             raise ValueError(f"Input features ({X.shape[1]}) does not match expected ({self.n_features_in_})")


        # 3. Convert data to PyTorch Tensor
        current_device = self._get_device()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(current_device)

        # 4. Perform Prediction
        self.model_.eval() # Set model to evaluation mode
        with torch.no_grad(): # Important: disable gradient calculation for inference
            outputs = self.model_(X_tensor)

        # 5. Convert back to NumPy array on CPU
        predictions = outputs.cpu().numpy()
        # Return 1D array as expected by scikit-learn regressors
        return predictions.flatten()

    # Optional but recommended for full scikit-learn compatibility & hyperparameter tuning
    def get_params(self, deep=True):
        """Gets parameters for this estimator."""
        # BaseEstimator provides a default implementation, but explicit is fine
        return {"lr": self.lr, "epochs": self.epochs, "batch_size": self.batch_size, "device": self.device}

    def set_params(self, **parameters):
        """Sets the parameters of this estimator."""
        # BaseEstimator provides a default implementation, but explicit is fine
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# -------------------------------------