import torchvision.transforms as T


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