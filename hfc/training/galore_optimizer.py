import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class GaLoreOptimizer(Optimizer):
    """
    Production-ready GaLore optimizer.
    """
    def __init__(
        self,
        params,
        base_optimizer_cls,
        lr: float = 1e-4,
        rank: int = 128,
        update_proj_gap: int = 200,
        scale_factor: float = 0.25,
        **kwargs,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not rank > 0:
            raise ValueError(f"Invalid rank: {rank}")
        
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale_factor = scale_factor

        self.base_optimizer_cls = base_optimizer_cls
        self.base_optimizer_kwargs = kwargs
        self.base_optimizer = None

        self.projection_matrices: Dict[int, Any] = {}
        self.global_step = 0
        self._is_hook_injected = False

        defaults = dict(lr=lr)
        super(GaLoreOptimizer, self).__init__(params, defaults)

    def _initialize_base_optimizer(self, initial_param):
        logger.info("Initializing base optimizer...")
        self.base_optimizer = self.base_optimizer_cls([initial_param], **self.base_optimizer_kwargs)
        self.base_optimizer.param_groups[0]['params'] = []
        logger.info(f"Base optimizer ({self.base_optimizer_cls.__name__}) initialized.")
    
    @torch.no_grad()
    def _update_projection_matrices(self, grad: torch.Tensor, param_id: int):
        if grad.dim() < 2:
            return

        grad_2d = grad.view(grad.shape[0], -1)
        m, n = grad_2d.shape
        
        if m >= n:
            _, _, Vh = torch.linalg.svd(grad_2d, full_matrices=False)
            P = Vh[:self.rank, :].T.contiguous()
            self.projection_matrices[param_id] = (P, 'right')
        else:
            U, _, _ = torch.linalg.svd(grad_2d, full_matrices=False)
            P = U[:, :self.rank].contiguous()
            self.projection_matrices[param_id] = (P, 'left')

    def _get_low_rank_grad(self, grad: torch.Tensor, param_id: int):
        if param_id not in self.projection_matrices:
            return None

        P, proj_type = self.projection_matrices[param_id]
        grad_2d = grad.view(grad.shape[0], -1)

        if proj_type == 'right':
            return grad_2d.T @ P
        else:
            return P.T @ grad_2d

    def _project_back(self, low_rank_update: torch.Tensor, param_id: int, original_shape):
        if param_id not in self.projection_matrices:
            raise ValueError("Projection matrix not found.")
        
        P, proj_type = self.projection_matrices[param_id]
        
        if proj_type == 'right':
            full_rank_update = P @ low_rank_update.T
        else:
            full_rank_update = P @ low_rank_update
            
        return full_rank_update.view(original_shape)

    def _create_hook(self, p):
        param_id = id(p)

        def hook(*args):
            if p.grad is None: return

            grad = p.grad.data
            
            if self.global_step % self.update_proj_gap == 1 and grad.dim() >= 2:
                self._update_projection_matrices(grad, param_id)

            if grad.dim() >= 2 and param_id in self.projection_matrices:
                low_rank_grad = self._get_low_rank_grad(grad, param_id)
                
                if param_id not in self.base_optimizer.state:
                    self.base_optimizer.state[param_id] = {}
                state = self.base_optimizer.state[param_id]
                
                dummy_param = nn.Parameter(torch.zeros_like(low_rank_grad))
                
                if not state:
                    self.base_optimizer.param_groups[0]['params'] = [dummy_param]
                    self.base_optimizer.step()
                    state.update(self.base_optimizer.state.pop(dummy_param))
                
                dummy_param.grad = low_rank_grad
                self.base_optimizer.param_groups[0]['params'] = [dummy_param]
                self.base_optimizer.state[dummy_param] = state
                self.base_optimizer.step()
                self.base_optimizer.state[param_id].update(self.base_optimizer.state.pop(dummy_param))

                low_rank_update = dummy_param.data
                update = self._project_back(low_rank_update, param_id, p.shape)
                p.add_(update, alpha=self.scale_factor)
            else:
                self.base_optimizer.param_groups[0]['params'] = [p]
                self.base_optimizer.step()
                # self.base_optimizer.state[p] = self.base_optimizer.state.pop(p)

            p.grad = None
        
        return hook

    def inject_hooks(self):
        logger.info("Injecting hooks for per-layer updates...")
        if self.base_optimizer is None:
            first_param = next((p for group in self.param_groups for p in group['params']), None)
            if first_param is None:
                raise ValueError("Optimizer has no parameters.")
            self._initialize_base_optimizer(first_param)

        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    p.register_post_accumulate_grad_hook(self._create_hook(p))
        self._is_hook_injected = True
        logger.info("Hooks injected successfully.")
    
    def step(self, closure=None):
        # This is called by Accelerator after grad accumulation
        # In hook mode, this mainly just increments the step counter
        self.global_step += 1
        # The actual parameter updates happen in the hooks
        if closure is not None:
            with torch.enable_grad():
                closure()

    def zero_grad(self, set_to_none: bool = True):
        # Grads are already set to None in hooks.
        # This function is still needed for the external API.
        pass
