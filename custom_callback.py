from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import update_learning_rate

class CustomCallback(BaseCallback):
    def __init__(self, save_freq, save_path, initial_lr, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.initial_lr = initial_lr

    # def _on_rollout_end(self):
        
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = f"{self.save_path}/model65_{self.n_calls}_steps"
            self.model.save(path)
            if self.verbose:
                print(f"Saved to {path}")

        # learning rate scheduler (progress_remaining goes from 1.0 → 0.0)
        progress_remaining = 1.0 - (self.num_timesteps / self.model._total_timesteps)
        new_lr = self.initial_lr * max(progress_remaining, 0.1)
        update_learning_rate(self.model.policy.optimizer, new_lr)
        self.logger.record("train/learning_rate", new_lr)

        # success rate monitoring
        infos = self.locals.get("infos", [])
        for info in infos:
            if "success_rate" in info:
                self.logger.record("train/success_rate", info["success_rate"])
        return True