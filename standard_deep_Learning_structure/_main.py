import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
csv_path = ''
for root, _, files in os.walk('.'):
    for name in files:
        if name == 'data.csv':
            csv_path = os.path.join(root, name)
assert csv_path != '', 'Could not locate the data.csv file'

dataset = pd.read_csv(csv_path, sep=';')
train_set, valid_set = train_test_split(dataset, test_size=0.25, random_state=0)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset
# objects
train_dl = ChallengeDataset(train_set, 'train')
valid_dl = ChallengeDataset(valid_set, 'val')

# create an instance of our ResNet model
net = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
criterion = t.nn.BCELoss()

# set up the optimizer (see t.optim)
optimizer = t.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)  # lr=0.001

# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(net, criterion, optimizer, train_dl, valid_dl, cuda=True, early_stopping_patience=8)

# go, go, go... call fit on trainer
res = trainer.fit(500)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
print("train_losses: \n", res[0])
print("\nval_losses:\n", res[1])
