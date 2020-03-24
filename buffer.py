from collections import deque, namedtuple
import random
import torch
import numpy as np

class PrioritizedExperience(object):
    """docstring for ."""

    def __init__(self, experience, priority):
        self.experience = experience
        self.priority = priority

class SimpleBuffer:

    def __init__(self, device, seed, hyperparams):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = [None] * hyperparams["buffer_size"]
        self.total_episodes = 0
        self.buffer_size = hyperparams["buffer_size"]
        self.batch_size = hyperparams["batch_size"]
        self.device = device
        self.seed = random.seed(seed)

    def add(self, experience):
        """Add a new experience to memory."""
        self.memory[self.total_episodes % self.buffer_size] = experience
        self.total_episodes = self.total_episodes + 1

    def update_priorities(self, indices, losses):
        pass

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        indices = np.random.choice(len(self), size=self.batch_size)

        states = torch.from_numpy(np.vstack([self.memory[idx].start_state \
            for idx in indices if self.memory[idx] is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([self.memory[idx].action \
            for idx in indices if self.memory[idx] is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([self.memory[idx].reward \
            for idx in indices if self.memory[idx] is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([self.memory[idx].next_state \
            if self.memory[idx].next_state is not None else self.memory[idx].start_state \
            for idx in indices if self.memory[idx] is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([self.memory[idx].done \
            for idx in indices if self.memory[idx] is not None]).astype(np.uint8)).float().to(self.device)
        weights = torch.from_numpy(np.vstack([1.0 \
            for idx in indices if self.memory[idx] is not None])).float().to(self.device)

        return states, actions, rewards, next_states, dones, weights, indices

    def __len__(self):
        """Return the current size of internal memory."""
        return min(self.total_episodes, self.buffer_size)

class PriorityBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, device, seed, hyperparams):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = [None] * hyperparams["buffer_size"]
        self.total_experiences = 0
        self.buffer_size = hyperparams["buffer_size"]
        self.batch_size = hyperparams["batch_size"]
        self.device = device
        self.seed = random.seed(seed)
        self.priority_epsilon = hyperparams["priority_epsilon"]
        self.alpha = hyperparams["priority_alpha"]
        self.beta_start = hyperparams["priority_beta_start"]
        self.beta_end = hyperparams["priority_beta_end"]
        self.beta_schedule_experiences = hyperparams["priority_beta_schedule_experiences"]

    def add(self, experience):
        """Add a new experience to memory."""
        if self.total_experiences > 0:
            max_priority = max(pe.priority for pe in self.memory if pe is not None)
        else:
            max_priority = 1
        self.memory[self.total_experiences % self.buffer_size] = PrioritizedExperience(experience, max_priority)
        self.total_experiences = self.total_experiences + 1

    def beta(self):
        return min(self.beta_end, self.beta_start + self.total_experiences * \
            (self.beta_end - self.beta_start) / self.beta_schedule_experiences)

    def update_priorities(self, indices, losses):
        unnormalized_probs = (losses + self.priority_epsilon) ** self.alpha
        for (index, replacement) in zip(indices, unnormalized_probs):
                self.memory[index].priority = replacement.item()

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        sum_priorities = sum(pe.priority for pe in self.memory if pe is not None)
        all_probs  = [pe.priority/sum_priorities for pe in self.memory if pe is not None]
        indices = np.random.choice(len(self), size=self.batch_size, p=all_probs)

        states = torch.from_numpy(np.vstack([self.memory[idx].experience.start_state \
            for idx in indices if self.memory[idx] is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([self.memory[idx].experience.action \
            for idx in indices if self.memory[idx] is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([self.memory[idx].experience.reward \
            for idx in indices if self.memory[idx] is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([self.memory[idx].experience.next_state \
            if self.memory[idx].experience.next_state is not None else self.memory[idx].experience.start_state \
            for idx in indices if self.memory[idx] is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([self.memory[idx].experience.done \
            for idx in indices if self.memory[idx] is not None]).astype(np.uint8)).float().to(self.device)
        probs = torch.from_numpy(np.vstack([self.memory[idx].priority / sum_priorities \
            for idx in indices if self.memory[idx] is not None])).float().to(self.device)

        weights = (len(self) * probs) ** -self.beta()
        weights /= max(weights)

        return states, actions, rewards, next_states, dones, weights, indices

    def __len__(self):
        """Return the current size of internal memory."""
        return min(self.total_experiences, self.buffer_size)
