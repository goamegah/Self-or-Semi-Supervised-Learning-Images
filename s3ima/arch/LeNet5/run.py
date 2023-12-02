import torch
import torch.backends.cudnn as cudnn
import torchvision
import matplotlib.pyplot as plt
import pickle
import sys
import os

sys.path.insert(0, "../../../")
# from arch
from s3ima.arch.LeNet5.model import LeNet5

# From local helper files

from helper_evaluation import set_all_seeds, compute_confusion_matrix, compute_accuracy
from helper_train import train_model
from helper_plotting import plot_training_loss, plot_accuracy, show_examples, plot_confusion_matrix
from helper_dataset import get_dataloaders_mnist

import argparse

model_names = ['ResNet18', 'LeNet5']

##########################
# SETTINGS
##########################

parser = argparse.ArgumentParser(description='PyTorch ResNet')

parser.add_argument('-m', '--mode',
                    metavar='MODE',
                    default='train',
                    help='which mode use during running model',
                    choices=['train', 'eval'])

parser.add_argument('-data',
                    metavar='DIR',
                    default='./data',
                    help='path to dataset')

parser.add_argument('-dn', '--dataset-name',
                    default='mnist',
                    help='dataset name',
                    choices=['cifar10', 'mnist'])

parser.add_argument('-a', '--arch',
                    metavar='ARCH',
                    default='LeNet5',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: LeNet5)')

parser.add_argument('-j', '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 32)')

parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size',
                    default=10,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('-eval-batch-size',
                    default=256,
                    type=int,
                    metavar='N',
                    help='Cause of having mini dataset, must be between 5 and 10')

parser.add_argument('--lr', '--learning-rate',
                    default=0.0003,
                    type=float,
                    metavar='LR',
                    help='initial learning rate', dest='lr')

parser.add_argument('--wd', '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--n-classes',
                    default=10,
                    type=int,
                    help='number of classes (default: 10)')

parser.add_argument('--seed',
                    default=1,
                    type=int,
                    help='seed for initializing training. ')

parser.add_argument('--disable-cuda',
                    action='store_true',
                    help='Disable CUDA')

parser.add_argument('--fp16-precision',
                    action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--log-every-n-steps',
                    default=10,
                    type=int,
                    help='Log every n steps')


def main():
    args = parser.parse_args()

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    # Other
    GRAYSCALE = True  # for MNIST dataset

    set_all_seeds(args.seed)
    # set_deterministic

    ##########################
    # ## MNIST DATASET
    ##########################

    resize_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((32, 32)),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5,), (0.5,))])

    train_loader, valid_loader, test_loader = get_dataloaders_mnist(args=args,
                                                                    validation_fraction=0.3,
                                                                    train_transforms=resize_transform,
                                                                    test_transforms=resize_transform)

    print('size of sample train dataset:', len(train_loader))
    # Checking the dataset
    for images, labels in train_loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        print('Class labels of 10 examples:', labels[:10])
        break

    model = LeNet5(grayscale=True, num_classes=10)
    model.to(device=args.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           factor=0.1,
                                                           mode='max',
                                                           verbose=True)

    if args.mode == 'train':
        # dict for saving results
        summary = {}

        # train arch
        minibatch_loss_list, train_acc_list, valid_acc_list = train_model(
            model=model,
            num_epochs=args.epochs,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            device=args.device,
            logging_interval=args.log_every_n_steps)

        # display training loss function
        plot_training_loss(minibatch_loss_list=minibatch_loss_list,
                           num_epochs=args.epochs,
                           iter_per_epoch=len(train_loader),
                           results_dir="./figures",
                           averaging_iterations=100)
        plt.show()

        plot_accuracy(train_acc_list=train_acc_list,
                      valid_acc_list=valid_acc_list,
                      results_dir='./figures')
        # plt.ylim([80, 100])
        plt.show()

        model.cpu()
        show_examples(model=model, data_loader=test_loader, results_dir='./figures')
        plt.show()

        # class used for confusion matrix axis ticks
        class_dict = {0: '0',
                      1: '1',
                      2: '2',
                      3: '3',
                      4: '4',
                      5: '5',
                      6: '6',
                      7: '7',
                      8: '8',
                      9: '9'}

        # Confusion matrix for testing arch
        mat = compute_confusion_matrix(model=model,
                                       data_loader=test_loader,
                                       device=torch.device('cpu'))
        plot_confusion_matrix(mat,
                              class_names=class_dict.values(),
                              results_dir='./figures')
        plt.show()

        summary['minibatch_loss_list'] = minibatch_loss_list
        summary['valid_acc_list'] = valid_acc_list
        summary['train_acc_list'] = train_acc_list
        summary['confusion_matrix'] = mat
        summary['num_epochs'] = args.epochs
        summary['iter_per_epoch'] = len(train_loader)
        summary['averaging_iterations'] = 100

        # Save trained arch for further usage
        os.makedirs("./saved_data", exist_ok=True)

        # save dictionary to person_data.pkl file
        with open('./saved_data/LeNet5_summary.pkl', 'wb') as fp:
            pickle.dump(summary, fp)
            print('dictionary saved successfully to file')

        torch.save(obj=model.state_dict(), f="saved_data/model.pt")
        torch.save(obj=optimizer.state_dict(), f="./saved_data/optimizer.pt")
        torch.save(obj=scheduler.state_dict(), f="./saved_data/scheduler.pt")

    # eval
    else:
        model.load_state_dict(state_dict=torch.load(f="./saved_data/model.pt"))
        # model is assume to be trained
        # optimizer.load_state_dict(state_dict=torch.load(f="saved_data/optimizer.pt"))

        test_acc = compute_accuracy(model, test_loader, device=args.device)
        print(f'Test accuracy {test_acc :.2f}%')


# main program
if __name__ == '__main__':
    main()
