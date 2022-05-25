import matplotlib.pyplot as plt
import numpy as np

def set_plt_settings():
    plt.rcParams.update({'font.size': 18})
    SMALL_SIZE = 10
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def show_image_batch(img_list, mean, std, title=None):
    num = len(img_list)
    fig = plt.figure(figsize=(30, 30))
    for i in range(num):
      inp = img_list[i].numpy().transpose((1, 2, 0))
      inp = std * inp + mean
      inp = np.clip(inp, 0, 1)
      ax = fig.add_subplot(1, num, i+1)
      ax.imshow(inp)
      ax.set_title(title[i])
      #hide x-axis
      ax.get_xaxis().set_visible(False)
      #hide y-axis 
      ax.get_yaxis().set_visible(False)

    plt.show()