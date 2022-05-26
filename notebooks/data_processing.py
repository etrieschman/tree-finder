import torchvision.transforms as T
from torch.utils.data import DataLoader, sampler
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

# constants
IMAGE_DIM = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def define_transforms(scale_up, crop, scale_out, mean, std, mirror=True, randomcrop=True):
  # define transformations
  transforms = {}
  transforms['orig'] = T.Compose([T.Resize(scale_up), T.CenterCrop(crop), 
                        T.Resize(scale_out),
                        T.ToTensor(),
                        T.Normalize(mean, std)])
  if mirror:
    transforms['mirr'] = T.Compose([T.Resize(scale_up), T.CenterCrop(crop), 
                          T.RandomHorizontalFlip(1), 
                          T.Resize(scale_out),
                          T.ToTensor(),
                          T.Normalize(mean, std)])

  # include a random cropping (flipped and unflipped) at three crop sizes
  if randomcrop:
    for pct in [10, 25, 50]:
      scale_param = int(1 / (pct*0.01))
      transforms[f'crop{pct}'] = T.Compose([T.Resize(scale_param*scale_up), 
                                            T.RandomResizedCrop(crop),
                                            T.Resize(scale_out),
                                            T.ToTensor(),
                                            T.Normalize(mean, std)])
      if mirror:       
        transforms[f'crop{pct}_mirr'] = T.Compose([T.Resize(scale_param*scale_up), 
                                                  T.RandomResizedCrop(crop),
                                                  T.RandomHorizontalFlip(1),
                                                  T.Resize(scale_out),
                                                  T.ToTensor(),
                                                  T.Normalize(mean, std)])
  
  return transforms


def train_val_test_dataset(dataset, test_split, val_split, seed):
    # split off test data first
    trainval_idx, test_idx = train_test_split(
      list(range(len(dataset))), test_size=test_split, random_state=seed)
    # split remaining train/val indexes
    train_idx, val_idx = train_test_split(
      list(range(len(trainval_idx))), test_size=val_split, random_state=seed)
    # map those indices back onto original dataset
    train_idx = [trainval_idx[i] for i in train_idx]
    val_idx = [trainval_idx[j] for j in val_idx]
    datasets = {
      'all': dataset,
      'train': Subset(dataset, train_idx),
      'validate': Subset(dataset, val_idx),
      'test': Subset(dataset, test_idx)
      }
    return datasets


def make_dataloaders(
  dataset, test_split=0.10, val_split=0.25, sampleN=4, batch_size=32, seed=None):
  
  N = len(dataset)
  # split into train/val
  datasets = train_val_test_dataset(dataset, test_split, val_split, seed)

  # make dataloaders
  dataloaders = {
    k:DataLoader(datasets[k], batch_size=batch_size, shuffle=True) 
    for k in ['all', 'train','validate','test']
    }

  # make dictionary of sizes
  ds_sizes = {
    k:len(datasets[k]) for k in ['all', 'train','validate','test']
  }
  
  # add on sample loader
  dataloaders['sampler'] = DataLoader(dataset=datasets['all'], batch_size=sampleN, shuffle=True)
  ds_sizes['sampler'] = sampleN
  print('dataloader sizes:', ds_sizes)

  return dataloaders, ds_sizes



  