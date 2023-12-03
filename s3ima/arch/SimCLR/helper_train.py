import logging
import os
import sys

import torch
from torch.utils.tensorboard import SummaryWriter
from helper_config import save_config_file, save_checkpoint


def train(model, optimizer,
          train_loader, valid_loader,
          test_loader, args,
          name='training',
          criterion=None):

    minibatch_loss_list, train_loss_list, train_acc_list, valid_loss_list, valid_acc_list = [], [], [], [], []

    writer = SummaryWriter(log_dir=f'{name}_runs')
    # config logging file
    logging.basicConfig(filename=os.path.join(writer.log_dir, f'{name}.log'),
                        level=logging.DEBUG)
    # save config file
    save_config_file(writer.log_dir, args)

    loss_hist_valid = [0] * args.train_epochs
    accuracy_hist_valid = [0] * args.train_epochs

    n_iter = 0

    for epoch in range(args.train_epochs):
        train_acc_epoch = 0
        train_loss_epoch = 0
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            is_correct = (torch.argmax(input=pred, dim=1) == y_batch).float()
            accuracy = is_correct.sum()

            train_loss_epoch += loss_value * y_batch.size(0)
            train_acc_epoch += accuracy
            minibatch_loss_list.append(loss_value)

            # Enregistrement à chaque itération
            writer.add_scalar(tag='train/loss', scalar_value=loss_value, global_step=n_iter)
            writer.add_scalar(tag='train/acc', scalar_value=accuracy, global_step=n_iter)
            n_iter += 1  # Mettre à jour le compteur global

        train_acc_list.append(train_acc_epoch/len(train_loader.dataset))
        train_loss_list.append(train_acc_epoch/len(train_loader.dataset))

        if valid_loader is not None:
            valid_acc_epoch = 0
            valid_loss_epoch = 0
            # Évaluation à chaque époque
            model.eval()
            with torch.no_grad():
                for x_batch, y_batch in valid_loader:
                    pred = model(x_batch)
                    loss = criterion(pred, y_batch)
                    is_correct = (torch.argmax(input=pred, dim=1) == y_batch).float()
                    valid_loss_epoch += loss.item() * y_batch.size(0)
                    valid_acc_epoch += is_correct.sum()

                valid_loss_epoch /= len(valid_loader.dataset)
                valid_acc_epoch /= len(valid_loader.dataset)

            loss_hist_valid[epoch] = valid_loss_epoch
            accuracy_hist_valid[epoch] = valid_acc_epoch

            valid_acc_list.append(valid_acc_epoch)
            valid_loss_list.append(valid_loss_epoch)

            writer.add_scalar(tag='valid/loss', scalar_value=loss_hist_valid[epoch], global_step=epoch)
            writer.add_scalar(tag='valid/acc', scalar_value=accuracy_hist_valid[epoch], global_step=epoch)

            print(f'Epoch {epoch +1} '
                  f'val_accuracy: {accuracy_hist_valid[epoch]: .4f} '
                  f'val_loss: {loss_hist_valid[epoch]: .4f} '
                  f'test data: {len(test_loader.dataset)} '
                  f'train data: {len(train_loader.dataset)} ')

    logging.info("Training has finished.")
    # save model checkpoints
    checkpoint_name = f'{args.arch}_train_{args.train_epochs:04d}.pth.tar'
    save_checkpoint(state={'epoch': args.train_epochs,
                           'arch': args.arch,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict(),
                           },
                    is_best=False,
                    filename=os.path.join(writer.log_dir, checkpoint_name))
    logging.info(f"Model checkpoint and metadata has been saved at {writer.log_dir}.")

    return minibatch_loss_list, train_acc_list, valid_acc_list
