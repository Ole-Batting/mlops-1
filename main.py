import argparse
import sys

import torch

from data import mnist
from model import MyAwesomeModel

import matplotlib.pyplot as plt

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set, test_set = mnist()
        
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        
        epochs = 30
        train_losses, test_losses = [], []
        for e in range(epochs):
            model.train()
            running_loss = 0
            steps = 0
            for images, labels in train_set:
                optimizer.zero_grad()
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()/images.shape[0]
                steps += 1
            else:
                with torch.no_grad():
                    model.eval()
                    images, labels = test_set
                    log_ps = model(images)
                    _, top_class = log_ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy = torch.mean(equals.type(torch.FloatTensor))
                    loss = criterion(log_ps, labels)/images.shape[0]
                    test_losses.append(loss)
                    print(f'{e}: train loss: {running_loss/steps}, test loss: {loss}, test accuracy: {accuracy}')
            train_losses.append(running_loss/steps)
            
        plt.figure(figsize=(8,4))
        plt.plot(train_losses, label='train')
        plt.plot(test_losses, label='test')
        plt.legend()
        plt.show()
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = torch.load(args.load_model_from)
        _, test_set = mnist()
        
        with torch.no_grad():
            model.eval()
            images, labels = test_set
            log_ps = model(images)
            _, top_class = log_ps.topk(1, dim=1)
            equals = tops_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            print(f'test accuracy: {accuracy}')

if __name__ == '__main__':
    TrainOREvaluate()
    

    
    
    
    
    
    
    
    