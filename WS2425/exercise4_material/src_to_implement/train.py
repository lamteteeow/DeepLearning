import torch as t
# from torch.optim.lr_scheduler import LambdaLR
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


## Initialization of weights ##


def initialize_weights(layer):
    if (layer is t.nn.Conv2d) or (layer is t.nn.Linear):
        t.nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            t.nn.init.uniform_(layer.bias, a=0.0, b=1.0)


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
data_csv = pd.read_csv("data.csv", sep=";")
train, test = train_test_split(data_csv, test_size=0.25, random_state=42)
# print(train)
# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO
train_dataset = ChallengeDataset(train, mode="train")
test_dataset = ChallengeDataset(test, mode="val")

train_dl = t.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_dl = t.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)

# create an instance of our ResNet model
# TODO
resnet = model.ResNet()

resnet = resnet.apply(initialize_weights)

# def get_lr(scheduler_epoch, lr_schedule):
#    for (start, end), lr_value in lr_schedule.items():
#        if start <= scheduler_epoch + 1 <= end:
#            return lr_value
#    return 1e-8

# lr_schedule = {(1, 5): 1e-3, (6, 20): 1e-4, (21, 70): 1e-5, (71, 1000): 1e-6}
# scheduler = LambdaLR(optimizer, lr_lambda=lambda scheduler_epoch: get_lr(scheduler_epoch, lr_schedule))

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
# TODO
criterion = t.nn.BCELoss()
optim = t.optim.Adam(resnet.parameters(), lr=5e-3)
resnet_trainer = Trainer(
    model=resnet,
    crit=criterion,
    optim=optim,
    train_dl=train_dl,
    val_test_dl=test_dl,
    cuda=True,
    # early_stopping_patience=-1,
    early_stopping_patience=35
)

# go, go, go... call fit on trainer
# TODO
res = resnet_trainer.fit(500)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label="train loss")
plt.plot(np.arange(len(res[1])), res[1], label="val loss")
plt.yscale("log")
plt.legend()
plt.savefig("losses.png")
