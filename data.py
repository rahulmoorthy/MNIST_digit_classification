import matplotlib
import numpy as np
import os
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from model import *
import matplotlib.pyplot as plt

# utility lib. for file path, mkdir,
minst_dataset_dir = os.path.join(os.getcwd(), 'minst_dataset')
label_file_path = os.path.join(minst_dataset_dir, 'label.txt')

data_list = []

with open(label_file_path, "r") as f:
    for line in f:
        if line.startswith('#'):
            continue
        tokens = line.split(';')  # split the line by ','
        label = int(tokens[0])
        file_path = tokens[-1].strip()  # .strip() removes space or empty char.
        data_list.append({'label': label, 'file_path': file_path})

random.shuffle(data_list)

total_items = len(data_list)

# Training, ratio: 60%
n_train_sets = 0.6 * total_items
train_set_list = data_list[: int(n_train_sets)]

# Validation, ratio: 20%
n_valid_sets = 0.2 * total_items
valid_set_list = data_list[int(n_train_sets): int(n_train_sets + n_valid_sets)]

# Testing, ratio: 20%
test_set_list = data_list[int(n_train_sets + n_valid_sets):]

class MINSTDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        #print("__len__ call")
        return len(self.data_list)

    def __getitem__(self, idx):
        #print("__getitem__ call")
        item = self.data_list[idx]
        label = np.asarray(item['label'])
        file_path = os.path.join(minst_dataset_dir, item['file_path'])

        # Load image as gray-scale image and convert to (0, 1)
        img = np.asarray(Image.open(file_path).convert('L'), dtype=np.float32) / 255.0
        h, w = img.shape[0], img.shape[1]

        # Create image tensor
        img_tensor = torch.from_numpy(img)

        # Reshape to (1, 28, 28), the 1 is the channel size
        img_tensor = img_tensor.view((1, h, w))
        label_tensor = torch.from_numpy(label).long()  # Loss measure require long type tensor

        return img_tensor, label_tensor

train_dataset = MINSTDataset(train_set_list)
train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=64,
                                                shuffle=True,
                                                num_workers=0)
print('Total training items', len(train_dataset), ', Total training mini-batches in one epoch:', len(train_data_loader))

valid_set = MINSTDataset(valid_set_list)
valid_data_loader = torch.utils.data.DataLoader(valid_set,
                                                batch_size=32,
                                                shuffle=True,
                                                num_workers=0)
print('Total validation set:', len(valid_set))

# Visulize the dataset.
# idx, (image, label) = next(enumerate(train_data_loader))  # we can use next(*) load once.
# print('image tensor shape (N, C, H, W):', image.shape)
# print('label tensor shape (N, labels):', label.shape)
#
# n_batch_size = image.shape[0]
# channels = image.shape[1]
# h,w = image.shape[2], image.shape[3]
#
# nd_img = image.cpu().numpy()
# nd_label = label.cpu().numpy()
#
# # We show 4 examples:
# figs, axes = plt.subplots(1, 4)
# for i in range(0, 4):
#     axes[i].imshow(nd_img[i].reshape(h, w), cmap='gray')
#     axes[i].set_title('Label:' + str(nd_label[i]))
# plt.show()

