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
  dataloaders, dataset_sizes, device, model_path, epochs=3):
    
    # track dictionary for best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # loop epochs
    acc_history = {'train':[], 'validate':[],'epoch_train':[], 'epoch_validate':[]}
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
                acc_history[phase].append((running_corrects / running_num).cpu())
                data_bar.set_description(
                    '{} epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}%'
                    .format(phase, epoch+1, epochs, running_loss / running_num, 
                            running_corrects / running_num * 100))

            # update line search decay
            if (phase == 'train') & (scheduler is not None):
                scheduler.step()

            # tracking for best model
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            acc_history[f'epoch_{phase}'].append(epoch_acc.cpu())
            if phase == 'validate' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), model_path + 'temp/best_current_model.pt')
    
    # load best model
    print(f'\nReturning best model, with validation accuracy {best_acc}')
    model.load_state_dict(best_model_wts)
    return acc_history
