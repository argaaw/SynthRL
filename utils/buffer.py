import torch

class Replaybuffer():
    def __init__(self, n_states: int, n_actions: int, capacity: int, device: str):
        self.actions = torch.zeros((n_states, capacity, n_actions), dtype=torch.int64).to(device)
        self.rewards = torch.ones((n_states, capacity, 1), dtype=torch.float32).to(device) * -10
        self.capacity = capacity

    def store(self, preset_UID: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor):
        min_rewards, min_inds = self.rewards[preset_UID].min(dim=1)
        update = (rewards > min_rewards).squeeze()
        update_uid = preset_UID[update]

        self.rewards[update_uid, min_inds[update].squeeze(-1)] = rewards[update]
        self.actions[update_uid, min_inds[update].squeeze(-1)] = actions[update]

    def sample(self, preset_UID: torch.Tensor):
        indices = torch.randint(0, self.capacity, (len(preset_UID), 1)).squeeze()
        return self.actions[preset_UID, indices], self.rewards[preset_UID, indices]

    def save(self, path):
        torch.save(self.actions, path.joinpath('actions.pt'))
        torch.save(self.rewards, path.joinpath('rewards.pt'))

    def load(self, path):
        self.actions = torch.load(path.joinpath('actions.pt'))
        self.rewards = torch.load(path.joinpath('rewards.pt'))
