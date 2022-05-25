import torchvision.models as models
import torch.nn as nn
import torch

class ResnetBinClassifier(nn.Module):
    def __init__(self, resnet50=False):
        super().__init__()

        # pick right resnet model
        if resnet50:
          self.model = models.resnet50(pretrained=True)
          print('loading pretrained resnet50...')
        else:
          self.model = models.resnet18(pretrained=True)
          print('loading pretrained resnet18...')
        
        # update final layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
          nn.Linear(num_ftrs, 128),
          nn.ReLU(inplace=True),
          nn.Linear(128, 2))

        # freeze all but last layer
        self.grad_state = {'unfrozen':[], 'frozen':[]}
        for name, child in self.model.named_children():
          if name in ['fc']:
            self.grad_state['unfrozen'] += [name]
            for param in child.parameters():
              param.requires_grad = True
          else:
            self.grad_state['frozen'] += [name]
            for param in child.parameters():
              param.requires_grad = False
    
    def get_grad_state(self):
      return self.grad_state

    def forward(self, x):
        out = self.model(x)
        return out
