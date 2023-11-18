import matplotlib.pyplot as plt
import argparse

import torch
import torchvision
import torch.backends.cudnn as cudnn
from torchvision import transforms
import sys

sys.path.insert(0, "../../../")
from s3ima.arch.ResNet18.model import ResNet18, BasicBlock
from s3ima.arch.ResNet18.helper_evaluation import set_all_seeds, compute_confusion_matrix

from helper_evaluation import compute_accuracy
from helper_train import train_model
from helper_dataset import get_dataloaders_mnist
from helper_plotting import plot_training_loss, plot_accuracy, show_examples, plot_confusion_matrix

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
                    default='./datasets',
                    help='path to dataset')

parser.add_argument('-dn', '--dataset-name',
                    default='mnist',
                    help='dataset name',
                    choices=['cifar10', 'mnist'])

parser.add_argument('-a', '--arch',
                    metavar='ARCH',
                    default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')

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
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

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

parser.add_argument('--n_classes',
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

    set_all_seeds(args.seed)

    # Other
    GRAYSCALE = True  # for MNIST dataset

    ##########################
    # MNIST DATASET
    ##########################

    # Note transforms.ToTensor() scales input images
    # to 0-1 range

    # CONSTRAINT: We assume having 100 samples available
    # get_dataloader takes it an account
    resize_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((32, 32)),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5,), (0.5,))])

    train_loader, valid_loader, test_loader = get_dataloaders_mnist(  # batch_size=batch_size,
        args=args,
        validation_fraction=0.3,
        train_transforms=resize_transform,
        test_transforms=resize_transform)

    print('size of sample train dataset:', len(train_loader))
    print('size of sample valid dataset:', len(valid_loader))
    print('size of sample test dataset:', len(test_loader))
    # Checking the dataset
    for images, labels in train_loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        print('Class labels of 10 examples:', labels[:10])
        break

    torch.manual_seed(args.seed)

    model = ResNet18(num_layers=18,
                     block=BasicBlock,
                     num_classes=10,
                     grayscale=GRAYSCALE)

    model.to(args.device)
    # print(model)

    # Total parameters and trainable parameters.
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"{total_params:,} total parameters.")
    # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"{total_trainable_params:,} training parameters.")

    # Passing Dummy Tensor
    ######################

    # tensor = torch.rand([1, 3, 224, 224])
    # output = model(tensor)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    if args.mode == 'train':

        minibatch_loss_list, train_acc_list, valid_acc_list = train_model(
            model=model,
            num_epochs=args.epochs,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            device=args.device,
            logging_interval=5
        )

        # display training loss function
        plot_training_loss(minibatch_loss_list=minibatch_loss_list,
                           num_epochs=args.epochs,
                           iter_per_epoch=len(train_loader),
                           results_dir=None,
                           averaging_iterations=args.log_every_n_steps)
        plt.show()

        plot_accuracy(train_acc_list=train_acc_list,
                      valid_acc_list=valid_acc_list,
                      results_dir=None)
        # plt.ylim([80, 100])
        plt.show()

        model.cpu()
        show_examples(model=model, data_loader=test_loader)

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

        mat = compute_confusion_matrix(model=model, data_loader=test_loader, device=torch.device('cpu'))
        print(mat)
        plot_confusion_matrix(mat, class_names=class_dict.values())
        plt.show()

        # save
        torch.save(obj=model.state_dict(), f="saved_data/model.pt")
        torch.save(obj=optimizer.state_dict(), f="saved_data/optimizer.pt")
        # torch.save(obj=scheduler.state_dict(),f="saved_data/scheduler.pt")

    else:  # eval mode
        model.load_state_dict(state_dict=torch.load(f="saved_data/model.pt"))
        # optimizer.load_state_dict(state_dict=torch.load(f="saved_data/optimizer.pt"))

        test_acc = compute_accuracy(model, test_loader, device=args.device)
        print(f'Test accuracy {test_acc :.2f}%')


if __name__ == '__main__':
    main()
