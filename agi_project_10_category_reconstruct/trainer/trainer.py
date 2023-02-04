import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import shutil
import pandas as pd


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_dl=None,  # Validation (or test) data set
                 cuda=True  # Whether to use the GPU
                 ):
        self._model = model
        self._crit = crit

        self._optim = optim
        self._lr = optim.state_dict()['param_groups'][0]['lr']

        self._train_dl = train_dl
        self._val_test_dl = val_dl
        self._cuda = cuda

        if self._cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, iteration, valid_accuracy):
        torch.save({'state_dict': self._model.state_dict()},
                   'checkpoints/checkpoint_{:04d}_{:.3f}.ckp'.format(iteration, valid_accuracy))

    def restore_checkpoint(self, checkpointfile):
        ckp = torch.load(checkpointfile, 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients
        self._model.zero_grad()

        # -propagate through the network
        predictions = self._model(x)

        # -calculate the loss
        # print('predictions:', predictions)
        y = torch.argmax(y, -1)
        # print('y:', y)
        loss = self._crit(predictions, y)

        # -compute gradient by backward propagation
        loss.backward()

        # -update weights
        if self._optim is not None:
            self._optim.step()

        # -return the loss
        return loss

    def val_test_step(self, x, y):

        # predict
        # propagate through the network and calculate the loss and predictions
        predictions = self._model(x)
        y = torch.argmax(y, -1)

        loss = self._crit(predictions, y)

        # return the loss and the predictions
        return loss, predictions

    def train_epoch(self):

        # set training mode
        self._model.train()
        # iterate through the training set
        train_dl = DataLoader(self._train_dl, batch_size=32, shuffle=True)

        loss = 0
        for x, y in tqdm(train_dl):
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                x = x.cuda()
                y = y.cuda()

            # perform a training step
            loss += self.train_step(x, y)
        # calculate the average loss for the epoch and return it

        return loss / len(train_dl)

    def val_test(self):

        # set eval mode
        self._model.eval()

        # disable gradient computation
        with torch.no_grad():

            # iterate through the validation set
            valid_dl = DataLoader(self._val_test_dl, batch_size=32, shuffle=True)

            # perform a validation step
            loss = 0

            pred_list = torch.empty(0, 12)
            y_list = torch.empty(0, 12)
            if self._cuda:
                pred_list = pred_list.cuda()
                y_list = y_list.cuda()

            for x, y in tqdm(valid_dl):
                # transfer the batch to the gpu if given
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()

                loss_batch, predictions = self.val_test_step(x, y)
                loss += loss_batch

                # save the predictions and the labels for each batch
                y_list = torch.cat((y_list, y), 0)
                pred_list = torch.cat((pred_list, predictions), 0)

        _, pred_list = torch.max(pred_list, 1)
        pred_list = torch.nn.functional.one_hot(pred_list)

        # calculate the average loss and average metrics of your choice. You might want to calculate these
        # metrics in designated functions
        loss = loss / len(valid_dl)
        accuracy = accuracy_score(y_list.cpu(), pred_list.cpu())

        # return the loss and print the calculated metrics
        return loss, accuracy

    def fit(self, epochs=-1):

        # create a list for the train and validation losses, and create a counter for the epoch
        train_losses = []
        valid_losses = []
        accuracies = []
        epoch_id = 0

        while True:

            epoch_id += 1
            print("\nepoch: ", epoch_id)

            if epoch_id % 20 == 0:
                for param_group in self._optim.param_groups:
                    if param_group['lr'] > self._lr * 0.05:
                        param_group['lr'] = param_group['lr'] * 0.5

            # stop by epoch number
            if epoch_id >= epochs:
                break
            # train for a epoch and then calculate the loss and metrics on the validation set
            train_loss = self.train_epoch()
            print("train loss = ", train_loss)

            valid_loss, accuracy = self.val_test()
            print("valid_loss = ", valid_loss)
            print("accuracy = ", accuracy)

            '''# append the losses to the respective lists
            train_losses = np.append(train_losses, train_loss.cpu().detach())
            valid_losses = np.append(valid_losses, valid_loss.cpu().detach())
            accuracies = np.append(accuracies, accuracy)'''

            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            self.save_checkpoint(epoch_id, accuracy)

            # check whether early stopping should be performed using the early stopping criterion and stop if so

    def fine_tune(self, epochs, checkpointfile, lr=0.0005):

        self.restore_checkpoint(checkpointfile)

        epoch_id = 0

        for param_group in self._optim.param_groups:
            param_group['lr'] = lr
            self._lr = lr

        while True:

            epoch_id += 1
            print("\nepoch: ", epoch_id)

            if epoch_id % 20 == 0:
                for param_group in self._optim.param_groups:
                    if param_group['lr'] > self._lr * 0.05:
                        param_group['lr'] = param_group['lr'] * 0.5

            # stop by epoch number
            if epoch_id >= epochs:
                break
            # train for a epoch and then calculate the loss and metrics on the validation set
            train_loss = self.train_epoch()
            print("train loss = ", train_loss)

            valid_loss, accuracy = self.val_test()
            print("valid_loss = ", valid_loss)
            print("accuracy = ", accuracy)

            '''# append the losses to the respective lists
            train_losses = np.append(train_losses, train_loss.cpu().detach())
            valid_losses = np.append(valid_losses, valid_loss.cpu().detach())
            accuracies = np.append(accuracies, accuracy)'''

            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            self.save_checkpoint(epoch_id, accuracy)

            # check whether early stopping should be performed using the early stopping criterion and stop if so

    def test(self, checkpointfile):

        self.restore_checkpoint(checkpointfile)
        # set eval mode
        self._model.eval()

        test_dl = DataLoader(self._val_test_dl, batch_size=32, shuffle=True)

        # disable gradient computation
        with torch.no_grad():

            # perform a testation step

            pred_list = torch.empty(0, 12)
            image_name_list = []
            if self._cuda:
                pred_list = pred_list.cuda()

            for x, image_name in tqdm(test_dl):
                # transfer the batch to the gpu if given
                if self._cuda:
                    x = x.cuda()

                # predict
                predictions = self._model(x)

                # save the predictions and the labels for each batch
                pred_list = torch.cat((pred_list, predictions), 0)

                image_name = list(image_name)
                image_name_list += image_name

        pred_list = pred_list.cpu()

        _, pred_label_list = torch.max(pred_list, 1)
        labels = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney', 'Crosswalk', 'Hydrant', 'Motorcycle', 'Other',
                  'Palm', 'Stair', 'Traffic Light']
        column = [
            'ImageName,Bicycle,Bridge,Bus,Car,Chimney,Crosswalk,Hydrant,Motorcycle,Other,Palm,Stair,Traffic,Light']
        csv_data = []

        for i in range(len(pred_label_list)):
            pred_label = pred_label_list[i]
            image_name = image_name_list[i]
            prediction = pred_list[i, :]

            prediction = prediction.numpy().tolist()
            prediction = ','.join([str(x) for x in prediction])

            csv_row = image_name + ',' + prediction
            csv_data.append(csv_row)

            pred_label = labels[pred_label]
            shutil.copy('dataset/test/' + image_name, 'dataset/test_pre/' + pred_label + '/' + image_name)

        test_csv = pd.DataFrame(columns=column, data=csv_data)
        test_csv.to_csv('results_test.csv', index=False, sep=',')


if __name__ == '__main__':
    from Model_VGG16 import VGGnet
    from data import CAPTCHADataset
    from Model_ResNet import ResNet

    train_dataset = pd.read_csv('data_train.csv', sep=';')
    train_dl = CAPTCHADataset(train_dataset, 'train')

    valid_dataset = pd.read_csv('data_valid.csv', sep=';')
    valid_dl = CAPTCHADataset(valid_dataset, 'valid')

    # net = VGGnet(fine_tuning=True, num_classes=12)
    net = ResNet(num_classes=12)

    # set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
    criterion = torch.nn.CrossEntropyLoss()

    # set up the optimizer (see t.optim)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

    # create an object of type Trainer and set its early stopping criterion
    trainer = Trainer(net, criterion, optimizer, train_dl, valid_dl, cuda=True)

    trainer.fit(100)
