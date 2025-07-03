from stable_baselines3.common.callbacks import BaseCallback

class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = f"{self.save_path}/model_{self.n_calls}_steps"
            self.model.save(path)
            if self.verbose:
                print(f"Saved model checkpoint to {path}")
        return True