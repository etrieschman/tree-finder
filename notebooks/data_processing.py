import torchvision.transforms as T
from torch.utils.data import DataLoader, sampler

# constants
IMAGE_DIM = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def define_transforms(scale_up, crop, scale_out, mean, std):
  # define transformations
  transforms = {}
  transforms['orig'] = T.Compose([T.Resize(scale_up), T.CenterCrop(crop), 
                        T.Resize(scale_out),
                        T.ToTensor(),
                        T.Normalize(mean, std)])
  transforms['mirr'] = T.Compose([T.Resize(scale_up), T.CenterCrop(crop), T.RandomHorizontalFlip(1), 
                        T.Resize(scale_out),
                        T.ToTensor(),
                        T.Normalize(mean, std)])

  # include a random cropping (flipped and unflipped) at three crop sizes
  for pct in [10, 25, 50]:
      scale_param = int(1 / (pct*0.01))
      transforms[f'crop{pct}'] = T.Compose([T.Resize(scale_param*scale_up), T.RandomResizedCrop(crop),
                                            T.Resize(scale_out),
                                            T.ToTensor(),
                                            T.Normalize(mean, std)])
      transforms[f'crop{pct}_mirr'] = T.Compose([T.Resize(scale_param*scale_up), T.RandomResizedCrop(crop),
                                                T.RandomHorizontalFlip(1),
                                                T.Resize(scale_out),
                                                T.ToTensor(),
                                                T.Normalize(mean, std)])
  
  return transforms


def make_dataloaders(dataset, pct_train, pct_val, sampleN):
  N = len(dataset)
  # make dictionary of sizes
  ds_sizes = {'all': N,
            'train': int(N * pct_train),
            'validate': int(N * pct_val),
            'test': N - int(N * (pct_train + pct_val)),
            'sample': sampleN}
  print('dataloader sizes:', ds_sizes)

  # make loaders
  dataloaders = {}
  dataloaders['all'] = DataLoader(dataset=dataset, batch_size=N, shuffle=False)
  dataloaders['train'] = DataLoader(dataset=dataset, batch_size=32,
                                    sampler=sampler.SubsetRandomSampler(range(ds_sizes['train'])))
  dataloaders['validate'] = DataLoader(dataset=dataset, batch_size=32,
                                      sampler=sampler.SubsetRandomSampler(
                                          range(ds_sizes['train'], ds_sizes['train']+ds_sizes['validate'])))
  dataloaders['test'] = DataLoader(dataset=dataset, batch_size=32,
                                  sampler=sampler.SubsetRandomSampler(
                                      range(ds_sizes['train']+ds_sizes['validate'], N)))
  dataloaders['sample'] = DataLoader(dataset=dataset, batch_size=sampleN, shuffle=True)

  return dataloaders, ds_sizes



  