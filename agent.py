import numpy as np
import random
from collections import namedtuple, deque
import importlib

import model
importlib.reload(model)

import buffer
importlib.reload(buffer)

import experience
importlib.reload(experience)

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, use_double_dqn, use_priority_queue, hyperparams):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.use_double_dqn = use_double_dqn

        self.eps = hyperparams["eps_start"]
        self.eps_end = hyperparams["eps_end"]
        self.eps_decay = hyperparams["eps_decay"]

        # Q-Network
        self.qnetwork_local = model.QNetwork(state_size, action_size, seed, hyperparams).to(device)
        self.qnetwork_target = model.QNetwork(state_size, action_size, seed, hyperparams).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=hyperparams["learning_rate"])

        # Replay memory
        self.memory = buffer.PriorityBuffer(device, seed, hyperparams) \
            if use_priority_queue else buffer.SimpleBuffer(device, seed, hyperparams)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.update_every = hyperparams["update_every"]
        self.batch_size = hyperparams["batch_size"]
        self.tau = hyperparams["tau"]

        self.gamma = hyperparams["gamma"]
        self.num_steps = hyperparams["num_steps"]

    def learn_episode(self, env, brain_name, max_t):
        source = experience.FirstAndLastExperienceSource( \
        experience.ExperienceSource(env, self, brain_name, self.num_steps, max_t), self.gamma)
        for exp in source:
            self.learn_experience(exp)
        self.episode_end()
        return source.get_score()

    def learn_experience(self, experience):

        # Save experience in replay memory
        self.memory.add(experience)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                priorities = self.learn_batch(experiences, self.gamma ** self.num_steps)

    def episode_end(self):
        self.eps = max(self.eps_end, self.eps_decay*self.eps)

    def act(self, state, is_training=True):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if not is_training or random.random() > self.eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn_batch(self, experiences, final_gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            final_gamma (float): discount factor for the end state
        """
        states, actions, rewards, next_states, dones, batch_weights, indices = experiences

        # Get max predicted Q values (for next states) from target model (using local/target model for action selection (in normal / double dqn) )
        if self.use_double_dqn:
            Q_actions_select = self.qnetwork_local(next_states).detach().argmax(1)
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, Q_actions_select.unsqueeze(1))
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (final_gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        losses =  (Q_targets - Q_expected) ** 2
        self.memory.update_priorities(indices, losses)
        loss = (batch_weights * losses).mean()

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
