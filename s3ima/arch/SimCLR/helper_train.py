import logging
import os
import sys

import torch
from torch.utils.tensorboard import SummaryWriter
from helper_config import save_config_file, save_checkpoint


def train(model, optimizer, scheduler,
          train_loader, valid_loader,
          test_loader, args,
          criterion=None):

    if args.train_mode == 'pretrained':


    # finetune mode: just train classifier
    else:
        writer = SummaryWriter(log_dir='finetune_runs')
        # config logging file
        logging.basicConfig(filename=os.path.join(writer.log_dir, 'finetune_training.log'),
                            level=logging.DEBUG)
        # save config file
        save_config_file(writer.log_dir, args)

        """
        writer = SummaryWriter(log_dir='finetune_runs')
        # config logging file
        logging.basicConfig(filename=os.path.join(writer.log_dir, 'finetune_training.log'), level=logging.DEBUG)
        # save config file
        save_config_file(writer.log_dir, args)

        loss_hist_train = [0] * args.train_epochs
        accuracy_hist_train = [0] * args.train_epochs
        loss_hist_valid = [0] * args.train_epochs
        accuracy_hist_valid = [0] * args.train_epochs

        logging.info(f"Start SimCLR fine-tuning training for {args.train_epochs} epochs.")
        logging.info(f"Training with gpu: {args.disable_cuda}.")

        for epoch in range(args.train_epochs):
            model.train()
            for x_batch, y_batch in train_loader:
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_hist_train[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(input=pred, dim=1) == y_batch).float()
                accuracy_hist_train[epoch] += is_correct.sum()

            loss_hist_train[epoch] /= len(train_loader.dataset)
            accuracy_hist_train[epoch] /= len(train_loader.dataset) 
            writer.add_scalar(tag='train/loss', scalar_value=loss_hist_train[epoch], global_step=epoch)
            writer.add_scalar(tag='train/acc', scalar_value=accuracy_hist_train[epoch], global_step=epoch)
            writer.add_scalar(tag='train/learning_rate', scalar_value=scheduler.get_lr()[0], global_step=epoch)

            if args.finetune_fractions is not None:
                # mode eval
                model.eval()
                with torch.no_grad():
                    for x_batch, y_batch in test_loader:
                        pred = model(x_batch)
                        loss = criterion(pred, y_batch)
                        loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                        is_correct = (torch.argmax(input=pred, dim=1) == y_batch).float()
                        accuracy_hist_valid[epoch] += is_correct.sum()
                loss_hist_valid[epoch] /= len(test_loader.dataset)
                accuracy_hist_valid[epoch] /= len(test_loader.dataset)

                writer.add_scalar(tag='valid/loss', scalar_value=loss_hist_valid[epoch], global_step=epoch)
                writer.add_scalar(tag='valid/acc', scalar_value=accuracy_hist_valid[epoch], global_step=epoch)

                print(f'Epoch {epoch +1} '
                      f'train_accuracy: {accuracy_hist_train[epoch]: .4f} '
                      f'val_accuracy: {accuracy_hist_valid[epoch]: .4f} '
                      f'test data: {len(test_loader.dataset)} '
                      f'train data: {len(train_loader.dataset)} ')
        
        """

        loss_hist_valid = [0] * args.train_epochs
        accuracy_hist_valid = [0] * args.train_epochs

        global_step = 0

        for epoch in range(args.train_epochs):
            model.train()

            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()

                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                is_correct = (torch.argmax(input=pred, dim=1) == y_batch).float()
                accuracy = is_correct.sum() / y_batch.size(0)

                # Enregistrement à chaque itération
                writer.add_scalar(tag='train/loss', scalar_value=loss_value, global_step=global_step)
                writer.add_scalar(tag='train/acc', scalar_value=accuracy, global_step=global_step)
                writer.add_scalar(tag='learning_rate', scalar_value=scheduler.get_lr()[0], global_step=global_step)

                global_step += 1  # Mettre à jour le compteur global

            if valid_loader is not None:

                # Évaluation à chaque époque
                model.eval()
                with torch.no_grad():
                    for x_batch, y_batch in valid_loader:
                        pred = model(x_batch)
                        loss = criterion(pred, y_batch)
                        loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                        is_correct = (torch.argmax(input=pred, dim=1) == y_batch).float()
                        accuracy_hist_valid[epoch] += is_correct.sum()
                loss_hist_valid[epoch] /= len(valid_loader.dataset)
                accuracy_hist_valid[epoch] /= len(valid_loader.dataset)

                writer.add_scalar(tag='valid/loss', scalar_value=loss_hist_valid[epoch], global_step=global_step)
                writer.add_scalar(tag='valid/acc', scalar_value=accuracy_hist_valid[epoch], global_step=global_step)

                print(f'Epoch {epoch +1} '
                      f'val_accuracy: {accuracy_hist_valid[epoch]: .4f} '
                      f'test data: {len(test_loader.dataset)} '
                      f'train data: {len(train_loader.dataset)} ')

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = f'{args.arch}_finetune_{args.train_epochs:04d}.pth.tar'
        save_checkpoint(state={'epoch': args.train_epochs,
                               'arch': args.arch,
                               'state_dict': model.state_dict(),
                               'optimizer': optimizer.state_dict(),
                               },
                        is_best=False,
                        filename=os.path.join(writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {writer.log_dir}.")

