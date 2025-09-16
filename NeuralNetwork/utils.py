from torch import nn
import random
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
from collections import OrderedDict

def create_random_network(input_shape,
                          maximum_width = 6, #number of neurons
                          maximum_depth = 2, #number of layers
                          activation_functions = [nn.ReLU()]):


    previous_width = random.randint(1, maximum_width)
    layers_dim = [previous_width]
    layers = [('hidden1', nn.Linear(input_shape, previous_width)),
                  ('act1', random.choice(activation_functions))]

    depth = random.randint(1, maximum_depth-1)

    for i in range(depth):

        layer_width = random.randint(1, maximum_width)
        layers_dim.extend([layer_width])

        layers.extend([(f'hidden{i+2}', nn.Linear(previous_width, layer_width)),
                       (f'act{i+2}', random.choice(activation_functions))])

        previous_width = layer_width

    layers.extend([(f'hidden{i+3}', nn.Linear(previous_width, 1))])

    return OrderedDict(layers), layers_dim

def create_network(input_shape,
                          width = 6, #number of neurons
                          depth = 2, #number of layers
                          activation_function = nn.ReLU()):


    layers = [('hidden1', nn.Linear(input_shape, width)),
                  ('act1', activation_function)]

    for i in range(depth):

        layers.extend([(f'hidden{i+2}', nn.Linear(width, width)),
                       (f'act{i+2}', activation_function)])

    layers.extend([(f'hidden{i+3}', nn.Linear(width, 1))])

    return OrderedDict(layers)

def _train_network(nn,
                      X_train, y_train,
                      X_val=None, y_val=None,
                      X_test=None, y_test=None,
                      epochs=100, batch_size=32, learning_rate=0.001,
                      criterion=nn.MSELoss(), optimizer=optim.Adam, device='cpu',
                      return_history =  False):


        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        do_validation = False
        if X_val is not None and y_val is not None:
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            do_validation = True
        do_test = False
        if X_test is not None and y_test is not None:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            do_test = True


        optimizer = optimizer(nn.model.parameters(), learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        # optim.lr_scheduler.OneCycleLR
        # optim.lr_scheduler.CosineAnnealingLR
        history = {'train_loss': [], 'val_loss': [], 'test_loss': []}

        for epoch in range(epochs):

            nn.model.train()
            running_loss = 0.0

            # Iterate through batches
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).view(-1, 1)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass - ensure inputs have correct shape
                outputs = nn.model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Calculate average training loss for the epoch
            epoch_train_loss = running_loss / len(train_loader)
            history['train_loss'].append(epoch_train_loss)

            # Validation step
            if do_validation:
                nn.model.eval()
                with torch.no_grad():
                    val_outputs = nn.model(X_val)
                    val_loss = criterion(val_outputs, y_val.view(-1, 1)).item()
                    history['val_loss'].append(val_loss)

                    # Learning rate scheduling based on validation loss
                    scheduler.step(val_loss)

            if do_test:
                nn.model.eval()
                with torch.no_grad():
                    test_outputs = nn.model(X_test)
                    test_loss = criterion(test_outputs, y_test.view(-1, 1)).item()
                    history['test_loss'].append(test_loss)

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                if do_validation:
                    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_train_loss:.4f}")


        if return_history:

            return history
        else:

            return nn

