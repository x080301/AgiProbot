from Model_VGG16 import VGGnet
import pandas as pd
from data import CAPTCHADataset
from Model_ResNet import ResNet
import torch
from trainer import Trainer

if __name__ == '__main__':
    '''# Train VGG19
    train_dataset = pd.read_csv('data_train.csv', sep=';')
    train_dl = CAPTCHADataset(train_dataset, 'train')

    valid_dataset = pd.read_csv('data_valid.csv', sep=';')
    valid_dl = CAPTCHADataset(valid_dataset, 'valid')

    net = VGGnet(fine_tuning=True, num_classes=12)
    # net = ResNet(num_classes=12)

    # set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
    criterion = torch.nn.CrossEntropyLoss()

    # set up the optimizer (see t.optim)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

    # create an object of type Trainer and set its early stopping criterion
    trainer = Trainer(net, criterion, optimizer, train_dl, valid_dl, cuda=True)

    trainer.fit(100)'''

    '''# Train ResNet
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

    trainer.fit(100)'''
    # VGG19 has better performance
    # get results using VGG19

    test_dataset = pd.read_csv('data_test.csv', sep=';')
    test_dl = CAPTCHADataset(test_dataset, 'test')

    net = VGGnet(fine_tuning=True, num_classes=12)
    # net = ResNet(num_classes=12)

    # set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
    criterion = torch.nn.CrossEntropyLoss()

    # set up the optimizer (see t.optim)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

    # create an object of type Trainer and set its early stopping criterion
    trainer = Trainer(net, criterion, optimizer, test_dl, test_dl, cuda=True)

    trainer.test('checkpoints/checkpoint_0088_0.693.ckp')
