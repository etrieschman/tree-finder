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

# resnet 50 transfer learning model, retraining last conv layer
class ResnetTransferClassifier(nn.Module):
    def __init__(self, num_classes=7, retrain_last_cnblock=False):
        super().__init__()

        self.model = models.resnet50(pretrained=True)
        
        # update final layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
          nn.Linear(num_ftrs, 128),
          nn.ReLU(inplace=True),
          nn.Linear(128, num_classes))

        if retrain_last_cnblock:
          retrain_layers = ['layer4', 'avgpool', 'fc']
        else:
          retrain_layers = ['avgpool', 'fc']

        # freeze all but last layer
        self.grad_state = {'unfrozen':[], 'frozen':[]}
        for name, child in self.model.named_children():
          if name in retrain_layers:
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


# convnext small, retraining last fully connected layer
class ConvnextTransferClassifier(nn.Module):
    def __init__(self, num_classes=7, retrain_last_cnblock=False):
        super().__init__()

        self.model = models.convnext_tiny(pretrained=True)
        
        # update final layer
        num_ftrs = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

        # freeze all but last layer
        self.grad_state = {'unfrozen':[], 'frozen':[]}
        for name, child in self.model.named_children():
          if name in ['avgpool', 'classifier']:
            self.grad_state['unfrozen'] += [name]
            for param in child.parameters():
              param.requires_grad = True
          else:
            self.grad_state['frozen'] += [name]
            for param in child.parameters():
              param.requires_grad = False
          
          # unfreeze last CNBlock if requested
          if name == 'features' and retrain_last_cnblock:
            self.grad_state['unfrozen'] += ['last cnn']
            for param in child.layers[-1].parameters():
              param.requires_grad = True
    
    def get_grad_state(self):
      return self.grad_state

    def forward(self, x):
        out = self.model(x)
        return out

# vision transformer
class TransformerTransferClassifier(nn.Module):
    def __init__(self, num_classes=7, retrain_last_encoder=False):
        super().__init__()

        self.model = models.vit_b_16(pretrained=True)
        
        # update final layer
        num_ftrs = self.model.heads[-1].in_features
        self.model.fc = nn.Sequential(
          nn.Linear(num_ftrs, 128),
          nn.ReLU(inplace=True),
          nn.Linear(128, num_classes))

        # freeze all but last layer
        self.grad_state = {'unfrozen':[], 'frozen':[]}
        for name, child in self.model.named_children():
          if name == 'heads':
            self.grad_state['unfrozen'] += [name]
            for param in child.parameters():
              param.requires_grad = True            
          else:
            self.grad_state['frozen'] += [name]
            for param in child.parameters():
              param.requires_grad = False
          # unfreeze last encoder if requested
          if (name == 'encoder') & retrain_last_encoder:
            self.grad_state['unfrozen'] += ['last encoder']
            for param in child.layers[-1].parameters():
              param.requires_grad = True
    
    def get_grad_state(self):
      return self.grad_state

    def forward(self, x):
        out = self.model(x)
        return out

