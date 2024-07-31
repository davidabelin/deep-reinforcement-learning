import wandb
import argparse
import numpy as np 
import random

# Training and evaluation demo code
def train_one_epoch(epoch, lr, bs): 
    acc = 0.25 + ((epoch/30) +  (random.random()/10))
    loss = 0.2 + (1 - ((epoch-1)/10 +  random.random()/5))
    return acc, loss

def evaluate_one_epoch(epoch): 
    acc = 0.1 + ((epoch/20) +  (random.random()/10))
    loss = 0.25 + (1 - ((epoch-1)/10 +  random.random()/6))
    return acc, loss


def main(args):
    # Start a W&B Run
    run = wandb.init(project="config_example", config=args)

    # Access values from config dictionary and store them into variables for readability
    lr  =  wandb.config['learning_rate']
    bs = wandb.config['batch_size']
    epochs = wandb.config['epochs']

    # Simulate training and logging values to W&B for demo purposes
    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(epoch, lr, bs)
        val_acc, val_loss = evaluate_one_epoch(epoch)

        wandb.log({
        'epoch': epoch, 
        'train_acc': train_acc,
        'train_loss': train_loss, 
        'val_acc': val_acc, 
        'val_loss': val_loss
        })

if __name__ == "__main__":
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=32,
    help="Batch size")
  
  parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=50,
    help="Number of training epochs")
  
  parser.add_argument(
    "-lr",
    "--learning_rate",
    type=int,
    default=0.001,
    help="Learning rate")    


args = parser.parse_args()

main(args)