import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from helper_config import save_config_file, save_checkpoint
from helper_evaluation import accuracy


def info_nce_loss(features, args):

    labels = torch.cat([torch.arange(args.batch_size) for _ in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(args.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(args.device)

    logits = logits / args.temperature
    return logits, labels


def train(model, optimizer,
          scheduler,
          train_loader,
          valid_loader,
          test_loader,
          args):

    if args.mode == 'eval':
        for epoch in range(args.epochs):
            top1_train_accuracy = 0
            for counter, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(args.device)
                y_batch = y_batch.to(args.device)

                logits = model(x_batch)
                loss = torch.nn.functional.cross_entropy(logits, y_batch)
                top1 = accuracy(logits, y_batch, topk=(1,))
                top1_train_accuracy += top1[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            top1_train_accuracy /= (counter + 1)
            top1_accuracy = 0
            top5_accuracy = 0
            for counter, (x_batch, y_batch) in enumerate(test_loader):
                x_batch = x_batch.to(args.device)
                y_batch = y_batch.to(args.device)

                logits = model(x_batch)

                top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
                top1_accuracy += top1[0]
                top5_accuracy += top5[0]

            top1_accuracy /= (counter + 1)
            top5_accuracy /= (counter + 1)
            print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\t"
                  f"Top1 Test accuracy: {top1_accuracy.item()}\t"
                  f"Top5 test acc: {top5_accuracy.item()}")
    else:
        scaler = GradScaler(enabled=args.fp16_precision)

        writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(writer.log_dir, 'training.log'), level=logging.DEBUG)

        # save config file
        save_config_file(writer.log_dir, args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {args.epochs} epochs.")
        logging.info(f"Training with gpu: {args.disable_cuda}.")

        for epoch_counter in range(args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(args.device)

                with autocast(enabled=args.fp16_precision):
                    features = model(images)
                    logits, labels = info_nce_loss(features=features, args=args)
                    loss = torch.nn.functional.cross_entropy(logits, labels)

                optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()

                if n_iter % args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    writer.add_scalar(tag='loss', scalar_value=loss, global_step=n_iter)
                    writer.add_scalar(tag='acc/top1', scalar_value=top1[0], global_step=n_iter)
                    writer.add_scalar(tag='acc/top5', scalar_value=top5[0], global_step=n_iter)
                    writer.add_scalar(tag='learning_rate', scalar_value=scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(args.epochs)
        save_checkpoint(state={'epoch': args.epochs,
                               'arch': args.arch,
                               'state_dict': model.state_dict(),
                               'optimizer': optimizer.state_dict(),
                               },
                        is_best=False,
                        filename=os.path.join(writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {writer.log_dir}.")

