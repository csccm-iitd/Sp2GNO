from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.optim as optim
import torch.nn as nn
import math


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_lr, T_max, eta_min=0):
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.T_max = T_max
        self.eta_min = eta_min
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        super(WarmupCosineAnnealingLR, self).__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.max_lr * (self.last_epoch + 1) / self.warmup_epochs for _ in self.base_lrs]
        else:
            self.cosine_scheduler.last_epoch = self.last_epoch - self.warmup_epochs
            return self.cosine_scheduler.get_lr()
    
    def step(self, epoch=None):
        if self.last_epoch < self.warmup_epochs:
            super(WarmupCosineAnnealingLR, self).step(epoch)
        else:
            self.cosine_scheduler.step(epoch - self.warmup_epochs)
"""# Example usage
model = nn.Linear(10, 1)  # Your model
optimizer = optim.SGD(model.parameters(), lr=0.1)

warmup_epochs = 10
max_lr = 0.1
T_max = 1000

scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs, max_lr, T_max)

# Training loop
num_epochs = 1500
for epoch in range(num_epochs):
    model.train()
    # Training code here...
    optimizer.step()
    scheduler.step()

    # Optionally, print the learning rate
    print(f'Epoch {epoch+1}/{num_epochs}, Learning Rate: {scheduler.get_lr()}')"""            
            




"""# Define the learning rate scheduler
T_0 = 100  # Number of epochs for the first restart
T_mult = 2  # Factor by which the number of epochs between restarts is multiplied
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=0)

# Training loop
num_epochs = 1500
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Step the scheduler
    scheduler.step(epoch)

    # Optionally, print the learning rate
    if epoch % 10 == 0:  # Print every 10 epochs
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{num_epochs}, Learning Rate: {current_lr}')"""            

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, base_lr, final_lr, after_scheduler=None, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.after_scheduler = after_scheduler
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.base_lr + (self.final_lr - self.base_lr) * (self.last_epoch / self.warmup_epochs) for _ in self.base_lrs]
        if self.after_scheduler:
            if self.last_epoch == self.warmup_epochs:
                self.after_scheduler.base_lrs = [self.final_lr for _ in self.base_lrs]
            return self.after_scheduler.get_lr()
        return [self.final_lr for _ in self.base_lrs]

    def step(self, epoch=None):
        if self.after_scheduler and self.last_epoch >= self.warmup_epochs:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
        else:
            return super(WarmupScheduler, self).step(epoch)
        
        
"""# Example usage
model = nn.Linear(10, 1)  # Your model
optimizer = optim.SGD(model.parameters(), lr=0.1)

warmup_epochs = 10
base_lr = 0.001
final_lr = 0.1
scheduler_after = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
scheduler = WarmupScheduler(optimizer, warmup_epochs, base_lr, final_lr, after_scheduler=scheduler_after)

# Training loop
num_epochs = 1500
for epoch in range(num_epochs):
    model.train()
    # Training code here...
    optimizer.step()
    scheduler.step(epoch)

    # Optionally, print the learning rate
    print(f'Epoch {epoch+1}/{num_epochs}, Learning Rate: {scheduler.get_last_lr()}')"""

def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return float(epoch) / float(max(1, warmup_epochs))
    return 1.0



class WarmupCosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult, eta_min, last_epoch)
        super(WarmupCosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [(self.base_lrs[i] * (self.last_epoch + 1) / self.warmup_epochs) for i in range(len(self.base_lrs))]
        else:
            return self.cosine_scheduler.get_lr()

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_epochs:
            super(WarmupCosineAnnealingWarmRestarts, self).step(epoch)
        else:
            self.cosine_scheduler.step(epoch - self.warmup_epochs if epoch is not None else None)
"""
model = nn.Linear(10, 1)  # Replace with your model
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Replace with your optimizer

warmup_epochs = 10  # Number of warmup epochs
T_0 = 100  # Number of epochs for the first restart
T_mult = 2  # Factor by which the number of epochs between restarts is multiplied
eta_min = 0.001  # Minimum learning rate after annealing

scheduler = WarmupCosineAnnealingWarmRestarts(optimizer, warmup_epochs, T_0, T_mult, eta_min)

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    # Your training code here
    optimizer.step()
    scheduler.step(epoch)

    # Optionally, print the learning rate
    if epoch % 10 == 0:  # Print every 10 epochs
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{num_epochs}, Learning Rate: {current_lr}')
"""
# Example usage



"""example usage
    
    1) 
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    
    2)
    
    warmup_epochs = 10
        max_lr = 0.1
        T_max = 1000

    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs, max_lr, T_max)
    
    
    3)
    scheduler_after = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    scheduler = WarmupScheduler(optimizer, warmup_epochs, base_lr, final_lr, after_scheduler=scheduler_after)
    
    
    """

