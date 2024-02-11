from typing import Union, Callable
import torch
from itertools import count
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
    This is the implementation of the SMART Loss function.
    However, the outputs of the model is not compatable with the original implementation 
    of the SMART Loss function, so we have to modify the original implementation to fit our.
    
    With n classification outputs (beside MLM), the model will have to load n times for each
    addition SMART loss, therefore making training process quite heavy, consum lots of VRAM.
    
    Note: With batch_size=32, max_length = 64 and 2 tasks, the model will consume around 
    10-11GB of VRAM.
"""

def kl_loss(input, target, reduction='batchmean'):
    return F.kl_div(
        F.log_softmax(input, dim=-1),
        F.softmax(target, dim=-1),
        reduction=reduction,
    )

def sym_kl_loss(input, target, reduction='sum', alpha=1.0):
    return alpha * F.kl_div(
        F.log_softmax(input, dim=-1),
        F.softmax(target.detach(), dim=-1),
        reduction=reduction,
    ) + F.kl_div(
        F.log_softmax(target, dim=-1),
        F.softmax(input.detach(), dim=-1),
        reduction=reduction,
    )

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d

def inf_norm(x):
    return torch.norm(x, p=float('inf'), dim=-1, keepdim=True)

class SMARTLoss(nn.Module):
    
    def __init__(
        self,
        eval_fn: Callable,
        loss_fn: Callable,
        loss_last_fn: Callable = None, 
        norm_fn: Callable = inf_norm, 
        num_steps: int = 1,
        step_size: float = 1e-3, 
        epsilon: float = 1e-6,
        noise_var: float = 1e-5,
    ) -> None:
        super().__init__()
        self.eval_fn = eval_fn 
        self.loss_fn = loss_fn
        self.loss_last_fn = default(loss_last_fn, loss_fn)
        self.norm_fn = norm_fn
        self.num_steps = num_steps 
        self.step_size = step_size
        self.epsilon = epsilon 
        self.noise_var = noise_var
        
    def forward(self, embed, state, input_mask,sent=True):
        noise = torch.randn_like(embed.float(), requires_grad=True) * self.noise_var
        
        if sent == True:
          # Indefinite loop with counter 
          for i in count():
            # Compute perturbed embed and states 
            embed_perturbed = embed + noise
            state_perturbed, _ = self.eval_fn(embed_perturbed.long(),input_mask)
            # Return final loss if last step (undetached state)
            if i == self.num_steps:
                return self.loss_last_fn(state_perturbed, state) 
            # Compute perturbation loss (detached state)
            loss = self.loss_fn(state_perturbed, state.detach())
            # Compute noise gradient ∂loss/∂noise
            noise_gradient, = torch.autograd.grad(loss, noise,allow_unused=True)
            # Move noise towards gradient to change state as much as possible 
            # Modify the computation of step to handle NoneType
            if noise_gradient is not None:
                step = noise + self.step_size * noise_gradient
            else:
                # Handle the case where noise_gradient is None, e.g., set step to noise
                step = noise
                
            
            # Normalize new noise step into norm induced ball 
            step_norm = self.norm_fn(step)
            noise = step / (step_norm + self.epsilon)
            # Reset noise gradients for next step
            noise = noise.detach().requires_grad_()
            
            
          
        else:
          for i in count():
              # Compute perturbed embed and states 
              embed_perturbed = embed + noise 
              _,state_perturbed = self.eval_fn(embed_perturbed.long(),input_mask)
              # Return final loss if last step (undetached state)
              if i == self.num_steps: 
                  return self.loss_last_fn(state_perturbed, state) 
              # Compute perturbation loss (detached state)
              loss = self.loss_fn(state_perturbed, state.detach())
              # Compute noise gradient ∂loss/∂noise
              noise_gradient, = torch.autograd.grad(loss, noise,allow_unused=True)
              if noise_gradient is not None:
                  step = noise + self.step_size * noise_gradient
              else:
                  # Handle the case where noise_gradient is None, e.g., set step to noise
                  step = noise  
              # Normalize new noise step into norm induced ball 
              step_norm = self.norm_fn(step)
              noise = step / (step_norm + self.epsilon)
              # Reset noise gradients for next step
              noise = noise.detach().requires_grad_()

class SMARTLoss1Label(nn.Module):
    
    def __init__(
        self,
        eval_fn: Callable,
        loss_fn: Callable,
        loss_last_fn: Callable = None, 
        norm_fn: Callable = inf_norm, 
        num_steps: int = 1,
        step_size: float = 1e-3, 
        epsilon: float = 1e-6,
        noise_var: float = 1e-5,
    ) -> None:
        super().__init__()
        self.eval_fn = eval_fn 
        self.loss_fn = loss_fn
        self.loss_last_fn = default(loss_last_fn, loss_fn)
        self.norm_fn = norm_fn
        self.num_steps = num_steps 
        self.step_size = step_size
        self.epsilon = epsilon 
        self.noise_var = noise_var
        
    def forward(self, embed, state, input_mask):
        noise = torch.randn_like(embed.float(), requires_grad=True) * self.noise_var
        
        
          # Indefinite loop with counter 
        for i in count():
            # Compute perturbed embed and states 
            embed_perturbed = embed + noise
            state_perturbed = self.eval_fn(embed_perturbed.long(),input_mask)
            # Return final loss if last step (undetached state)
            if i == self.num_steps:
                return self.loss_last_fn(state_perturbed, state) 
            # Compute perturbation loss (detached state)
            loss = self.loss_fn(state_perturbed, state.detach())
            # Compute noise gradient ∂loss/∂noise
            noise_gradient, = torch.autograd.grad(loss, noise,allow_unused=True)
            # Move noise towards gradient to change state as much as possible 
            # Modify the computation of step to handle NoneType
            if noise_gradient is not None:
                step = noise + self.step_size * noise_gradient
            else:
                # Handle the case where noise_gradient is None, e.g., set step to noise
                step = noise
                
            
            # Normalize new noise step into norm induced ball 
            step_norm = self.norm_fn(step)
            noise = step / (step_norm + self.epsilon)
            # Reset noise gradients for next step
            noise = noise.detach().requires_grad_()
            
class SMARTLoss3Label(nn.Module):
    
    def __init__(
        self,
        eval_fn: Callable,
        loss_fn: Callable,
        loss_last_fn: Callable = None, 
        norm_fn: Callable = inf_norm, 
        num_steps: int = 1,
        step_size: float = 1e-3, 
        epsilon: float = 1e-6,
        noise_var: float = 1e-5,
    ) -> None:
        super().__init__()
        self.eval_fn = eval_fn 
        self.loss_fn = loss_fn
        self.loss_last_fn = default(loss_last_fn, loss_fn)
        self.norm_fn = norm_fn
        self.num_steps = num_steps 
        self.step_size = step_size
        self.epsilon = epsilon 
        self.noise_var = noise_var
        
    def forward(self, embed, state, input_mask,task='sent'):
        noise = torch.randn_like(embed.float(), requires_grad=True) * self.noise_var
        
        if task == 'sent':
          # Indefinite loop with counter 
          for i in count():
            # Compute perturbed embed and states 
            embed_perturbed = embed + noise
            state_perturbed, _, _ = self.eval_fn(embed_perturbed.long(),input_mask)
            # Return final loss if last step (undetached state)
            if i == self.num_steps:
                return self.loss_last_fn(state_perturbed, state) 
            # Compute perturbation loss (detached state)
            loss = self.loss_fn(state_perturbed, state.detach())
            # Compute noise gradient ∂loss/∂noise
            noise_gradient, = torch.autograd.grad(loss, noise,allow_unused=True)
            # Move noise towards gradient to change state as much as possible 
            # Modify the computation of step to handle NoneType
            if noise_gradient is not None:
                step = noise + self.step_size * noise_gradient
            else:
                # Handle the case where noise_gradient is None, e.g., set step to noise
                step = noise
                
            
            # Normalize new noise step into norm induced ball 
            step_norm = self.norm_fn(step)
            noise = step / (step_norm + self.epsilon)
            # Reset noise gradients for next step
            noise = noise.detach().requires_grad_()
            
            
          
        elif task == 'clas':
          for i in count():
              # Compute perturbed embed and states 
              embed_perturbed = embed + noise 
              _,state_perturbed,_  = self.eval_fn(embed_perturbed.long(),input_mask)
              # Return final loss if last step (undetached state)
              if i == self.num_steps: 
                  return self.loss_last_fn(state_perturbed, state) 
              # Compute perturbation loss (detached state)
              loss = self.loss_fn(state_perturbed, state.detach())
              # Compute noise gradient ∂loss/∂noise
              noise_gradient, = torch.autograd.grad(loss, noise,allow_unused=True)
              if noise_gradient is not None:
                  step = noise + self.step_size * noise_gradient
              else:
                  # Handle the case where noise_gradient is None, e.g., set step to noise
                  step = noise  
              # Normalize new noise step into norm induced ball 
              step_norm = self.norm_fn(step)
              noise = step / (step_norm + self.epsilon)
              # Reset noise gradients for next step
              noise = noise.detach().requires_grad_()
        else:
          for i in count():
              # Compute perturbed embed and states 
              embed_perturbed = embed + noise 
              _,_,state_perturbed  = self.eval_fn(embed_perturbed.long(),input_mask)
              # Return final loss if last step (undetached state)
              if i == self.num_steps: 
                  return self.loss_last_fn(state_perturbed, state) 
              # Compute perturbation loss (detached state)
              loss = self.loss_fn(state_perturbed, state.detach())
              # Compute noise gradient ∂loss/∂noise
              noise_gradient, = torch.autograd.grad(loss, noise,allow_unused=True)
              if noise_gradient is not None:
                  step = noise + self.step_size * noise_gradient
              else:
                  # Handle the case where noise_gradient is None, e.g., set step to noise
                  step = noise  
              # Normalize new noise step into norm induced ball 
              step_norm = self.norm_fn(step)
              noise = step / (step_norm + self.epsilon)
              # Reset noise gradients for next step
              noise = noise.detach().requires_grad_()