# Playing around with the weights of the trained networks
import matplotlib.pyplot as plt
import os
import psutil
import torch

# PATH = '/home/karolis/k/goal-misgeneralization/train-procgen/logs/train/maze_pure_yellowline/maze-5x5/2023-02-09__23-01-32__seed_7766/'
PATH = '/home/karolis/k/goal-misgeneralization/train-procgen/logs/train/maze_pure_yellowline/maze-5x5/2023-02-10__09-48-21__seed_8998'  # the long training one
DEVICE = 'cpu'  # cpu/cuda
MAX_FILES = 5  # max files to load


def get_models(path):
    files = [f for f in os.listdir(path) if f.endswith('.pth')]
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    models = []
    device = torch.device(DEVICE)
    for f in files[:5]:
        model = torch.load(os.path.join(path, f), map_location=device)
        model['name'] = f
        models.append(model)
    return models


def main():
    print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)  # print RAM usage by the process
    models = get_models(PATH)
    print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)  # print RAM usage by the process
    w = []
    for model in models:
        w.append(model['model_state_dict']['embedder.block1.conv.weight'][0][0][0][0].item())
    plt.plot(w)
    plt.show()
    print()


if __name__ == '__main__':
    main()
