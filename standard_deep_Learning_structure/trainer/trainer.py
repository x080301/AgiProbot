import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients
        self._model.zero_grad()

        # -propagate through the network
        predictions = self._model(x)

        # -calculate the loss
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

        loss = self._crit(predictions, y)

        # return the loss and the predictions
        return loss, predictions

    def train_epoch(self):

        # set training mode
        self._model.train()
        # iterate through the training set
        train_dl = t.utils.data.DataLoader(self._train_dl, batch_size=32, shuffle=True)

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
        with t.no_grad():

            # iterate through the validation set
            valid_dl = t.utils.data.DataLoader(self._val_test_dl, batch_size=32, shuffle=True)

            # perform a validation step
            loss = 0
            pred_list = []
            y_list = []
            for x, y in tqdm(valid_dl):
                # transfer the batch to the gpu if given
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()

                loss_batch, predictions = self.val_test_step(x, y)
                loss += loss_batch

                # save the predictions and the labels for each batch
                y_list = np.append(y_list, y.cpu())

                pred_list = np.append(pred_list, np.around(predictions.cpu()))

        # calculate the average loss and average metrics of your choice. You might want to calculate these
        # metrics in designated functions
        loss = loss / len(valid_dl)
        avg_metric = f1_score(y_list, pred_list, average='weighted')
        print("f1_score = ", avg_metric)

        # return the loss and print the calculated metrics
        return loss, avg_metric

    def fit(self, epochs=-1):

        assert self._early_stopping_patience > 0 or epochs > 0

        # create a list for the train and validation losses, and create a counter for the epoch
        train_losses = []
        valid_losses = []
        f1_scores = []
        epoch_id = 0
        epoch_counter = 0  # counts continuous epochs when valid_loss dose not decrease.
        min_valid_loss_sum_3_epoches = 100

        while True:
            print("\nepoch: ", epoch_id)

            '''
            if epoch_id % 25 == 0:
                for param_group in self._optim.param_groups:
                    if param_group['lr']>0.0001:
                        param_group['lr'] = param_group['lr'] * 0.5
            '''

            # stop by epoch number
            if epoch_id >= epochs:
                break

            # train for a epoch and then calculate the loss and metrics on the validation set
            train_loss = self.train_epoch()
            print("train loss = ", train_loss)
            valid_loss, F1_Score = self.val_test()
            print("valid_loss = ", valid_loss)

            # append the losses to the respective lists
            train_losses = np.append(train_losses, train_loss.cpu().detach())
            valid_losses = np.append(valid_losses, valid_loss.cpu().detach())
            f1_scores=np.append(f1_scores, F1_Score)

            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            self.save_checkpoint(epoch_id)

            # check whether early stopping should be performed using the early stopping criterion and stop if so

            if epoch_id >= 4:

                valid_loss_sum_3_epoches = sum(valid_losses[epoch_id - 2:epoch_id + 1])
                print("min:  ", min_valid_loss_sum_3_epoches)
                print(("3 epoches:  ", valid_loss_sum_3_epoches))
                if valid_loss_sum_3_epoches >= min_valid_loss_sum_3_epoches:
                    epoch_counter += 1
                    if epoch_counter >= self._early_stopping_patience:
                        print("\nEarly Stop")
                        print("f1_scores : \n", f1_scores)
                        return train_losses, valid_losses
                else:
                    min_valid_loss_sum_3_epoches = valid_loss_sum_3_epoches
                    epoch_counter = 0

                print("counter: ", epoch_counter)

            epoch_id += 1

        # return the losses for both training and validation
