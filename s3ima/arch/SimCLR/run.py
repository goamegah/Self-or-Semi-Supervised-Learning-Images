import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from torch.utils.data import DataLoader
from helper_train import train
from helper_evaluation import compute_accuracy
from helper_dataset import SimclrDataset
from basemodel import ResNet18SimCLR
from s3ima.arch.ResNet18.model import ResNet18, BasicBlock

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

##########################
# SETTINGS
##########################

parser = argparse.ArgumentParser(description='PyTorch SimCLR')

parser.add_argument('-d', '--data',
                    metavar='DIR',
                    default='./datasets',
                    help='path to dataset')

parser.add_argument('-m', '--mode',
                    default='train',
                    help='Whether to perform training or evaluation.',
                    choices=['train', 'eval', 'train_then_eval'])

parser.add_argument('-tm', '--train-mode',
                    default='pretrain',
                    help='The train mode controls different objectives and trainable components.',
                    choices=['pretrain', 'finetune'])

parser.add_argument('-dn', '--dataset-name',
                    default='mnist',
                    help='dataset name',
                    choices=['stl10', 'cifar10', 'mnist'])

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

parser.add_argument('-e', '--epochs',
                    default=1,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size',
                    default=256,
                    type=int,
                    metavar='N',
                    help='train batch size.')


parser.add_argument('-eval-batch-size',
                    default=5,
                    help='The test batch size to use during evaluation '
                         'mode (must be less or equal min(train, valid, test) size.')

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

parser.add_argument('--out_dim',
                    default=128,
                    type=int,
                    help='feature dimension (default: 128)')

parser.add_argument('--log-every-n-steps',
                    default=10,
                    type=int,
                    help='Log every n steps')

parser.add_argument('--temperature',
                    default=0.07,
                    type=float,
                    help='softmax temperature (default: 0.07)')

parser.add_argument('--n-views',
                    default=2,
                    type=int,
                    metavar='N',
                    help='Number of views for contrastive learning training.')

parser.add_argument('--gpu-index',
                    default=0,
                    type=int,
                    help='Gpu index.')


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = SimclrDataset(root=args.data)

    if args.mode == "eval":
        torch.manual_seed(args.seed)
        train_ds, valid_ds, test_ds = dataset.get_dataset(args.dataset_name, args.n_views, args)
        train_loader = DataLoader(dataset=train_ds,
                                  batch_size=args.eval_batch_size,
                                  num_workers=0,
                                  drop_last=False,
                                  shuffle=False)
        valid_loader = DataLoader(dataset=valid_ds,
                                  batch_size=args.eval_batch_size,
                                  num_workers=0,
                                  drop_last=False,
                                  shuffle=False)
        test_loader = DataLoader(dataset=test_ds,
                                 batch_size=args.eval_batch_size,
                                 num_workers=0,
                                 drop_last=False,
                                 shuffle=False)

        # instance model
        model = ResNet18(num_layers=18,
                         block=BasicBlock,
                         num_classes=10,
                         grayscale=True)
        model.to(device=args.device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

        # load SimCLR saved model
        checkpoint = torch.load(f='/home/kgamegah/Documents/academic/UPC/deep-learning/ssima/ssima/arch/SimCLR/runs/'
                                  'Nov05_08-35-08_kgamegah-KLVD-WXX9/checkpoint_0100.pth.tar',
                                map_location=args.device)
        state_dict = checkpoint['state_dict']

        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    # remove prefix
                    state_dict[k[len("backbone."):]] = state_dict[k]
            del state_dict[k]

        # print(state_dict.keys())

        log = model.load_state_dict(state_dict, strict=False)
        assert log.missing_keys == ['fc.weight', 'fc.bias']

        # freeze all layers without the last fc
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False

        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == 2  # fc.weight, fc.bias

        train(model=model, optimizer=optimizer, scheduler=None,
              train_loader=train_loader, valid_loader=valid_loader,
              test_loader=test_loader, args=args)

        test_acc = compute_accuracy(model=model, data_loader=test_loader, device=args.device)
        print(f'Test accuracy {test_acc :.2f}%')

    else:  # train

        model = ResNet18SimCLR(projection_dim=args.out_dim)
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

        # if pretrain model
        # train, None, None
        # else
        # train, valid, test dataset
        train_dataset, _, _ = dataset.get_dataset(args.dataset_name, args.n_views, args)

        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=None,
                                  num_workers=args.workers,
                                  pin_memory=True,
                                  drop_last=True)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=len(train_loader),
                                                               eta_min=0,
                                                               last_epoch=-1)

        #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
        with torch.cuda.device(args.gpu_index):
            train(model=model, optimizer=optimizer, scheduler=scheduler,
                  train_loader=train_loader, valid_loader=None, test_loader=None, args=args)


if __name__ == "__main__":
    main()
