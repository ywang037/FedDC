
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# from scipy.linalg import sqrtm
import matplotlib.pyplot as plt


# generic purpose utils
def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def get_logger(logger_path):
    logging.basicConfig(
        filename=logger_path,
        # filename='/home/qinbin/test.log',
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%m-%d %H:%M', 
        level=logging.DEBUG, 
        filemode='w'
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    return logger 

# training and testing related utils
def evaluation(net, eval_loader, device):
    ''' Use this function to evalute the loss and accuracy of a model
    '''

    net.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
    correct, loss = 0, 0.0
    n_test_samples = len(eval_loader.dataset)
    
    with torch.no_grad():
        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            logits = net(x)
            pred = F.log_softmax(logits, dim=1)
            correct += (pred.argmax(dim=1) == y).sum().item()
            loss += criterion(pred,y).item()

    acc, loss = correct / n_test_samples, loss / n_test_samples    
    return acc, loss

def plot_series(
    series, 
    y_min=0, 
    y_max=1.0, 
    series_name='series_name', 
    title='Test acc', 
    step=1, 
    save_path=None
):
    """ input series must be a 1-d numpy array, e.g., the mean accuracy time series
    """
    series_len = series.shape[-1]
    colors = ['crimson', 'gold', 'deepskyblue', 'limegreen', 'deeppink', 'darkorange', 'sienna', 'slategrey']
    linestyles = ['-', '--', ':', '-.']
    
    run_colors, run_linestyles = colors[0], linestyles[0]
    plt.figure(figsize=(6,4),dpi=200)
    # steps = torch.tensor(range(1,series_len+1)).tolist()
    # data_x = [x for j, x in enumerate(steps) if (j+1) % step == 0]
    data_y = [y for j, y in enumerate(series) if (j+1) % step==0]
    plt.plot(data_y,label=series_name, linestyle=run_linestyles, color=run_colors, linewidth=1)
    plt.legend(loc='lower right')
        
    plt.grid()
    # plt.tight_layout()
    plt.xlim(1,series_len+1)
    # plt.xticks([int(i+1) for i in range(series_len)])
    plt.ylim(y_min,y_max)
    plt.xlabel('Comm. rounds')
    plt.ylabel(title, size='large')

    if save_path is not None:
        plt.savefig(save_path)