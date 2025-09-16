import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class UnifiedModel(nn.Module):
    def __init__(self, individual):
        super(UnifiedModel, self).__init__()
        self.main_net = individual[0]
        self.blocks = individual[1:]
        self.optimizer = None
        self.criterion = None

    def forward(self, x, test=False):
        if test:
            self.main_net.create_hidden_input(x, test=True)
            hidden_inputs = self.main_net.hidden_test_inputs
        else:
            self.main_net.create_hidden_input(x, test=False)
            hidden_inputs = self.main_net.hidden_inputs

        main_output = self.main_net(x)
        block_outputs = [
            block.forward(x, test=test, new_hidden_data=hidden_inputs)
            for block in self.blocks
        ]
        return main_output + sum(block_outputs)

    def compile(self, optimizer_cls=torch.optim.Adam, lr=0.001, criterion=nn.MSELoss()):
        self.optimizer = optimizer_cls(self.parameters(), lr=lr)
        self.criterion = criterion

    def backward_step(self, x, y):
        self.train()
        self.optimizer.zero_grad()
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y.view(-1, 1))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate(self, x_val, y_val):
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(x_val, test=True)
            val_loss = self.criterion(y_pred, y_val.view(-1, 1)).item()
        return val_loss

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):


        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            running_loss = 0.0



            for xb, yb in train_loader:
                loss = self.backward_step(xb, yb)
                running_loss += loss

            avg_train_loss = running_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)

            if X_val is not None and y_val is not None:
                val_loss = self.validate(X_val, y_val)
                history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

        return history
