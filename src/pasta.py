"""
PASTA: Policy-optimization via Adaptive Smooth Tchebycheff Attention

This module implements the PASTA algorithm, a multi-objective reinforcement 
learning (MORL) agent that uses adaptive smooth Tchebycheff scalarization 
to dynamically balance competing objectives based on gradient conflict.

Project Website: https://alejandromllo.github.io/research/pasta/

Reference:
    Murillo-González, A., Ali, M., & Liu, L. (2026). "Adaptive Smooth Tchebycheff 
    Attention for Multi-Objective Policy Optimization." Proceedings of Robotics: 
    Science and Systems (RSS), Sydney, Australia.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.distributions import Normal, Categorical


# ==========================================
# 1. ACTORS (Continuous & Discrete)
# ==========================================

class BaseActor(nn.Module):
    """
    Base network module for Actor classes.
    Processes the concatenated state and preference vector (w) through a shared multi-layer perceptron.
    """

    def __init__(self, state_dim, n_objectives, hidden_dim=64):
        super().__init__()
        self.input_dim = state_dim + n_objectives
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
    
    def forward(self, state, w):
        raise NotImplementedError

class ContinuousActor(BaseActor):

    def __init__(self, state_dim, action_dim, n_objectives, hidden_dim=64):
        super().__init__(state_dim, n_objectives, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

    def forward(self, state, w):
        if w.dim() == 1: w = w.unsqueeze(0).expand(state.size(0), -1)
        x = self.net(torch.cat([state, w], dim=1))
        mu = torch.sigmoid(self.mu_head(x))
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def get_action_logprob(self, state, w, deterministic=False):
        dist = self.forward(state, w)
        action = dist.mean if deterministic else dist.sample()
        return action, dist.log_prob(action).sum(dim=-1)

    def get_dist_eval(self, state, action, w):
        dist = self.forward(state, w)
        return dist.log_prob(action).sum(dim=-1), dist.entropy().sum(dim=-1)

class DiscreteActor(BaseActor):

    def __init__(self, state_dim, action_dim, n_objectives, hidden_dim=64):
        super().__init__(state_dim, n_objectives, hidden_dim)
        self.logits_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, w):
        if w.dim() == 1: w = w.unsqueeze(0).expand(state.size(0), -1)
        x = self.net(torch.cat([state, w], dim=1))
        return Categorical(logits=self.logits_head(x))

    def get_action_logprob(self, state, w, deterministic=False):
        dist = self.forward(state, w)
        if deterministic:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action)

    def get_dist_eval(self, state, action, w):
        dist = self.forward(state, w)
        if action.ndim > 1: action = action.squeeze(-1)
        return dist.log_prob(action), dist.entropy()

# ==========================================
# 2. CRITIC
# ==========================================

class Critic(nn.Module):

    def __init__(self, state_dim, n_objectives, hidden_dim=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + n_objectives, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)) 
            for _ in range(n_objectives)
        ])

    def forward(self, state, w):
        if w.dim() == 1: w = w.unsqueeze(0).expand(state.size(0), -1)
        features = self.trunk(torch.cat([state, w], dim=1))
        return torch.cat([head(features) for head in self.heads], dim=1)

# ==========================================
# 3. ROLLOUT BUFFER
# ==========================================

class RolloutBuffer:
    """
    Stores transitions during environment interaction for on-policy updates.
    Handles multi-objective rewards and computes Generalized Advantage Estimation (GAE) _per_ objective.
    """

    def __init__(self, n_steps, state_dim, n_objectives, gamma, gae_lambda, device):
        self.n_steps = n_steps; self.state_dim = state_dim; self.n_objectives = n_objectives
        self.gamma = gamma; self.gae_lambda = gae_lambda; self.device = device
        self.reset()

    def reset(self):
        self.states = np.zeros((self.n_steps, self.state_dim), dtype=np.float32)
        self.actions = None 
        self.log_probs = np.zeros((self.n_steps,), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.n_objectives), dtype=np.float32)
        self.values = np.zeros((self.n_steps + 1, self.n_objectives), dtype=np.float32)
        self.advantages = np.zeros((self.n_steps, self.n_objectives), dtype=np.float32)
        self.returns = np.zeros((self.n_steps, self.n_objectives), dtype=np.float32)
        self.dones = np.zeros((self.n_steps,), dtype=np.float32)
        self.ptr = 0; self.indices = np.arange(self.n_steps)

    def store(self, state, action, log_prob, reward, value, done):
        if self.actions is None:
            act_arr = np.array(action)
            shape = (self.n_steps,) if act_arr.ndim == 0 else (self.n_steps, *act_arr.shape)
            self.actions = np.zeros(shape, dtype=np.float32)
        if self.ptr < self.n_steps:
            self.states[self.ptr] = state; self.actions[self.ptr] = action
            self.log_probs[self.ptr] = log_prob; self.rewards[self.ptr] = reward
            self.values[self.ptr] = value; self.dones[self.ptr] = done; self.ptr += 1

    def compute_advantages_and_returns(self, last_value, last_done):
        self.values[self.ptr] = last_value
        gae = np.zeros(self.n_objectives, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            next_non_terminal = 1.0 - (last_done if t == self.n_steps - 1 else self.dones[t + 1])
            next_values = last_value if t == self.n_steps - 1 else self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae; self.returns[t] = gae + self.values[t]

    def get_batches(self, batch_size):
        np.random.shuffle(self.indices)
        for start in range(0, self.n_steps, batch_size):
            end = start + batch_size
            idx = self.indices[start:end]
            yield (
                torch.as_tensor(self.states[idx], device=self.device),
                torch.as_tensor(self.actions[idx], device=self.device),
                torch.as_tensor(self.log_probs[idx], device=self.device),
                torch.as_tensor(self.advantages[idx], device=self.device),
                torch.as_tensor(self.returns[idx], device=self.device),
                torch.as_tensor(idx, device=self.device).long()
            )

# ==========================================
# 4. Adaptive STCH Smoothness Controller
# ==========================================

class SmoothnessController:
    """
    Smoothness Controller for Adaptive Smooth Tchebycheff Scalarization function based on
    objective conflict ratio.

    Args:
        min_mu (float): The absolute minimum value the smoothness parameter can reach.
        max_mu (float): The starting and maximum value for the smoothness parameter.
        conflict_threshold (float): The ratio of conflicting gradients (0.0 to 1.0) above which the dynamic brake engages.
        ema_alpha (float): Exponential Moving Average (EMA) smoothing factor for mu updates.
        total_steps (int): Total expected training steps, used to calculate the base decay schedule.
        enable_decay (bool): If True, mu linearly decays from max_mu to min_mu over total_steps.
        enable_conflict (bool): If True, dynamically increases mu when the objective conflict ratio exceeds the threshold.

    Attributes:
        mu (float): The current operational smoothness parameter applied to the scalarization function.

    References:
        Murillo-González, A., Ali, M., & Liu, L. (2026). "Adaptive Smooth Tchebycheff Attention 
        for Multi-Objective Policy Optimization." Proceedings of Robotics: Science and Systems (RSS).
        Website: https://alejandromllo.github.io/research/pasta/
    """

    def __init__(self, 
                 min_mu=0.05, 
                 max_mu=10.0,          
                 conflict_threshold=0.4, 
                 ema_alpha=0.05, 
                 total_steps=10000,
                 enable_decay=True, 
                 enable_conflict=True):
        
        self.mu = max_mu
        self.start_mu = max_mu 
        
        self.min_mu = min_mu
        self.max_mu = max_mu
        self.total_steps = total_steps
        
        self.enable_decay = enable_decay
        self.enable_conflict = enable_conflict
        
        self.ema_alpha = ema_alpha
        self.conflict_threshold = conflict_threshold

    def update(self, current_step, conflict_ratio, return_details=False):

        target_mu = self.start_mu

        # 1. Decay Schedule (Target moves from Max -> Min)
        if self.enable_decay:
            progress = min(current_step / self.total_steps, 1.0)
            target_mu = self.start_mu - (self.start_mu - self.min_mu) * progress

        if return_details:
            # Save the "Base Schedule" for visualization before modifying it
            base_schedule_mu = target_mu

        # 2. Conflict Boost (Dynamic Brake)
        if self.enable_conflict and conflict_ratio > self.conflict_threshold:
            # Normalize the excess conflict between [Threshold, 1.0] -> [0.0, 1.0]
            # e.g. if ratio is 0.7 and threshold is 0.4, excess is 0.3 / 0.6 = 0.5 boost
            denom = 1.0 - self.conflict_threshold
            boost_factor = (conflict_ratio - self.conflict_threshold) / (denom + 1e-8)
            
            # Interpolate target back towards Max Mu based on conflict severity
            target_mu = target_mu + (self.max_mu - target_mu) * boost_factor

        # 3. EMA Update
        # mu_new = (1 - alpha) * mu_old + alpha * target
        self.mu = (1.0 - self.ema_alpha) * self.mu + self.ema_alpha * target_mu
        
        # Hard clamps for safety
        self.mu = max(self.min_mu, min(self.max_mu, self.mu))
        
        if return_details:
            return self.mu, target_mu, base_schedule_mu
        else:
            return self.mu


# ==========================================
# 5. PASTA: Policy-optimization via Adaptive Smooth Tchebycheff Attention
# ==========================================

class PASTA:
    """
    Policy-optimization via Adaptive Smooth Tchebycheff Attention (PASTA).

    This agent implements a variant of Proximal Policy Optimization (PPO) that incorporates 
    Adaptive Smooth Tchebycheff Attention mechanisms for improved and stable non-convex Pareto
    tradeoffs.

    Args:
        state_dim (int): Dimension of the observation space.
        action_dim (int): Dimension of the action space.
        n_objectives (int): Number of objectives in the multi-objective environment.
        device (torch.device): Target device for tensors (e.g., 'cpu' or 'cuda').
        continuous_actions (bool): Set to True for continuous action spaces, False for discrete.
        lr (float): Learning rate for both the actor and critic networks.
        n_steps (int): Number of steps to run per environment before an update.
        batch_size (int): Minibatch size for the optimization epochs.
        n_epochs (int): Number of optimization passes over the buffer per update.
        gamma (float): Discount factor for future rewards.
        gae_lambda (float): Bias vs variance trade-off factor for Generalized Advantage Estimation.
        clip_range (float): Clipping parameter for the PPO surrogate loss.
        vf_coef (float): Weight for the value function loss.
        ent_coef (float): Weight for the entropy bonus to encourage exploration.
        maintenance_rate_rho (float): Parameter blending Tchebycheff weights with uniform weights to prevent objective starvation.
        fixed_mu_value (float, optional): If provided, bypasses the adaptive controller and uses this static mu value.
        min_mu (float): Minimum mu limit for the internal Smoothness Controller.
        max_mu (float): Maximum/Start mu limit for the internal Smoothness Controller.
        moving_average_lambda (float): EMA coefficient for smoothing mu updates.
        conflict_ratio_kappa (float): Threshold for gradient conflicts to trigger mu adaptation.
        total_train_steps (int): Total expected environment steps for the decay schedule.
        enable_decay (bool): Toggle for linear decay in the Smoothness Controller.
        enable_conflict (bool): Toggle for conflict-based adaptation in the Smoothness Controller.

    Attributes:
        w (torch.Tensor): Current preference vector across objectives.
        global_min (torch.Tensor): Running minimum observed returns per objective for normalization.
        global_max (torch.Tensor): Running maximum observed returns per objective for normalization.

    References:
        Murillo-González, A., Ali, M., & Liu, L. (2026). "Adaptive Smooth Tchebycheff Attention 
        for Multi-Objective Policy Optimization." Proceedings of Robotics: Science and Systems (RSS).
        Website: https://alejandromllo.github.io/research/pasta/
    """

    def __init__(self, 
                 # Base Args
                 state_dim, action_dim, n_objectives, device,
                 continuous_actions=True, 
                 lr=3e-4, n_steps=2048, batch_size=64, n_epochs=10, 
                 gamma=0.99, gae_lambda=0.95, clip_range=0.2, vf_coef=0.5, ent_coef=0.01,
                 
                 # Adaptive & Ablation Flags
                 maintenance_rate_rho=0.15,
                 fixed_mu_value=None,       # If set, overrides the smoothness controller
                 
                 # Smoothness Controller Params
                 min_mu=0.05,
                 max_mu=10.0,
                 moving_average_lambda=0.05,
                 conflict_ratio_kappa=0.4,
                 total_train_steps=10000,
                 enable_decay=True,
                 enable_conflict=True
        ):
        
        self.device = device; self.n_objectives = n_objectives
        self.n_steps = n_steps; self.batch_size = batch_size
        self.n_epochs = n_epochs; self.clip_range = clip_range
        self.vf_coef = vf_coef; self.ent_coef = ent_coef
        
        self.w = torch.ones(n_objectives, dtype=torch.float32, device=device) / n_objectives
        self.global_min = torch.full((n_objectives,), float('inf'), device=device)
        self.global_max = torch.full((n_objectives,), float('-inf'), device=device)

        # Actor Factory
        ActorClass = ContinuousActor if continuous_actions else DiscreteActor
        self.actor = ActorClass(state_dim, action_dim, n_objectives).to(device)
        
        # Critic Factory
        self.critic = Critic(state_dim, n_objectives).to(device)
        
        # Buffer
        self.buffer = RolloutBuffer(n_steps, state_dim, n_objectives, gamma, gae_lambda, device)
        
        # Optimization
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.current_step = 0 
        
        # Internal Controller
        self.sm_controller = SmoothnessController(
            min_mu=min_mu,
            max_mu=max_mu,
            total_steps=total_train_steps,
            enable_decay=enable_decay,
            enable_conflict=enable_conflict,
            ema_alpha=moving_average_lambda,
            conflict_threshold=conflict_ratio_kappa
        )
        
        self.maintenance_rate = maintenance_rate_rho
        self.fixed_mu_value = fixed_mu_value
        self.utopia_point = 1.05
        self.last_conflict = 0.0

    def set_preference(self, w_numpy):
        self.w = torch.tensor(w_numpy, dtype=torch.float32, device=self.device)

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob = self.actor.get_action_logprob(state_t, self.w, deterministic=deterministic)
            value = self.critic(state_t, self.w)
        return action.cpu().numpy()[0], log_prob.item(), value.cpu().numpy()[0]

    def _update_stats(self, batch_returns):
        batch_min = torch.min(batch_returns, dim=0)[0]
        batch_max = torch.max(batch_returns, dim=0)[0]
        self.global_min = torch.minimum(self.global_min, batch_min)
        self.global_max = torch.maximum(self.global_max, batch_max)

    def pc_grad_update(self, objectives_grads):

        grads = [g.clone() for g in objectives_grads]
        pc_grads = [g.clone() for g in objectives_grads]
        random.shuffle(grads) 
        conflict_count = 0
        total_checks = 0

        for i in range(len(grads)):
            for j in range(len(grads)):
                if i == j: continue
                total_checks += 1
                g_i = pc_grads[i]; g_j = grads[j]
                dot = torch.dot(g_i, g_j)
                if dot < 0:
                    conflict_count += 1
                    denom = torch.dot(g_j, g_j) + 1e-8
                    pc_grads[i] -= (dot / denom) * g_j

        return pc_grads, conflict_count / max(1, total_checks)

    def _get_stch_scalar_utility(self, norm_data, mu):

        regrets = self.utopia_point - norm_data
        arg = (regrets * self.w) / mu
        return -mu * torch.logsumexp(arg, dim=1)

    def update_parameters(self):
        """
        Executes the main PASTA PPO update cycle:
        1. Updates normalization statistics based on recent buffer returns.
        2. Steps the Smoothness Controller to adapt 'mu' based on recent gradient conflicts.
        3. Computes Adaptive Smooth Tchebycheff Attention weights.
        4. Optimizes the Actor and Critic networks over multiple epochs.
        
        Returns:
            dict: Training statistics including value_loss, kl divergence, and current mu.
        """

        # 1. Update Stats & Normalize
        flat_returns = torch.tensor(self.buffer.returns, dtype=torch.float32, device=self.device)
        flat_values = torch.tensor(self.buffer.values[:-1], dtype=torch.float32, device=self.device)
        self._update_stats(flat_returns)
        
        range_vals = torch.clamp(self.global_max - self.global_min, min=1e-6)
        norm_returns = (flat_returns - self.global_min) / range_vals
        norm_values = (flat_values - self.global_min) / range_vals
        
        # 2. Update Mu
        if self.fixed_mu_value is not None:
            mu = self.fixed_mu_value
        else:
            mu = self.sm_controller.update(self.current_step, self.last_conflict)
        self.current_step += self.n_steps

        # 3. Calculate Attention Weights
        with torch.no_grad():
            regrets = self.utopia_point - norm_returns
            stch_weights = torch.softmax((regrets * self.w) / mu, dim=1)
            uniform = torch.ones_like(stch_weights) / self.n_objectives
            effective_weights = (1 - self.maintenance_rate) * stch_weights + \
                                (self.maintenance_rate) * uniform

        # 4. Update Loop
        avg_loss_vf = 0.0; kls = []; batch_conflicts = []
        
        for _ in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                states, actions, old_log_probs, _, vector_returns, idxs = batch
                
                # --- A. ASTCH Attention-Weighted Critic ---
                values = self.critic(states, self.w)
                batch_alpha = effective_weights[idxs].detach()                  # ASTCH Attention
                loss_v = (values - vector_returns).pow(2) * batch_alpha
                value_loss = self.vf_coef * loss_v.sum(dim=1).mean()
                
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                avg_loss_vf += value_loss.item()
                
                # --- B. Actor ---
                vec_adv = torch.tensor(self.buffer.advantages[idxs.cpu().numpy()], device=self.device)
                vec_adv = (vec_adv - vec_adv.mean(dim=0)) / (vec_adv.std(dim=0) + 1e-8)
                
                new_log_probs, entropy = self.actor.get_dist_eval(states, actions, self.w)
                ratio = (new_log_probs - old_log_probs).exp()
                
                grads_per_obj = []
                for obj_i in range(self.n_objectives):
                    self.actor_optimizer.zero_grad()
                    adv_i = vec_adv[:, obj_i]
                    surr1 = ratio * adv_i
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_i
                    (-torch.min(surr1, surr2).mean()).backward(retain_graph=True)
                    g_vec = [p.grad.view(-1) for p in self.actor.parameters() if p.grad is not None]
                    if g_vec: 
                        grads_per_obj.append(torch.cat(g_vec))
                
                if grads_per_obj:
                    proj_grads, c_ratio = self.pc_grad_update(grads_per_obj)
                    batch_conflicts.append(c_ratio)
                    
                    final_grad = sum(proj_grads)

                    self.actor_optimizer.zero_grad()
                    idx = 0
                    for param in self.actor.parameters():
                        n = param.numel()
                        if idx + n <= final_grad.shape[0]:
                            param.grad = final_grad[idx:idx+n].view(param.shape)
                        idx += n
                    
                    (-self.ent_coef * entropy.mean()).backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.actor_optimizer.step()

                with torch.no_grad(): 
                    kls.append((old_log_probs - new_log_probs).mean().item())

        self.last_conflict = np.mean(batch_conflicts) if batch_conflicts else 0.0
        return {"value_loss": avg_loss_vf / self.n_epochs, "kl": np.mean(kls), "mu": mu}
