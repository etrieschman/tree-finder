from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import copy

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)


def train_model(
  model, criterion, optimizer, scheduler, 
  dataloaders, dataset_sizes, device, epochs=3):
    
    # track dictionary for best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # loop epochs
    acc_history = {'train':[], 'validate':[]}
    for epoch in range(epochs):
        # run training and validating modes
        for phase in ['train', 'validate']:
            model.train() if (phase == 'train') else model.eval()

            running_num, running_loss, running_corrects = 0.0, 0.0, 0.0
            data_bar = tqdm(dataloaders[phase])
            # loop batches
            for inputs, labels in data_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # estimate and print summary stats
                _, preds = torch.max(outputs, 1)
                running_num += inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                acc_history[phase].append(running_corrects / running_num)
                data_bar.set_description(
                    '{} epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}%'
                    .format(phase, epoch, epochs, running_loss / running_num, 
                            running_corrects / running_num * 100))

            # update line search decay
            if phase == 'train':
              scheduler.step()

            # tracking for best model
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'validate' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    # load best model
    print(f'Returning best model, with validation accuracy {best_acc}')
    model.load_state_dict(best_model_wts)
    return model, acc_history

# def train_model(model, optimizer, loss_fn, epochs=1):
#     model = model.to(device=device)  # move the model parameters to CPU/GPU
#     for e in tqdm(range(epochs)):
#         for t, (x, y) in enumerate(loader_train):
#             model.train()  # put model to training mode
#             x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
#             y = y.to(device=device, dtype=torch.long)
            
#             scores = model(x)
#             loss = loss_fn # instantiate the passed loss function
#             loss = loss(scores, y)

#             # Zero out all of the gradients for the variables which the optimizer
#             # will update.
#             optimizer.zero_grad()

#             # This is the backwards pass: compute the gradient of the loss with
#             # respect to each  parameter of the model.
#             loss.backward()

#             # Actually update the parameters of the model using the gradients
#             # computed by the backwards pass.
#             optimizer.step()

#             if (t % print_every == 0):
#                 # print(f'Iteration {t}, loss = {loss.item():0.4}')
#                 acc = check_accuracy(loader_val, model)
#     return acc