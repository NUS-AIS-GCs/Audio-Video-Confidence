"""Optional Weights & Biases logging. Everything here is a no-op unless tracking is switched on.

Setup (Colab): put your key in Colab Secrets as WANDB_API_KEY and export it before running, e.g.
    os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
Optional: WANDB_ENTITY (your username or team), WANDB_PROJECT (default "avalidate").
"""
import os
from pathlib import Path

PROJECT = os.environ.get("WANDB_PROJECT", "avalidate")
ENTITY = os.environ.get("WANDB_ENTITY") or None


class Tracker:
    def __init__(self, enabled: bool, run_dir: Path, run_id: str, config: dict = None, resume: bool = False):
        self.run = None
        if not enabled:
            return
        import wandb
        id_file = run_dir / "wandb_run_id.txt"
        kwargs = dict(project=PROJECT, entity=ENTITY, group="audio", tags=["audio", "baseline"])
        if resume and id_file.exists():
            self.run = wandb.init(id=id_file.read_text().strip(), resume="allow", **kwargs)
        else:
            self.run = wandb.init(name=run_id, config=config, **kwargs)
            id_file.write_text(self.run.id)

    def log(self, data: dict, step: int = None) -> None:
        if self.run:
            self.run.log(data, step=step)

    def summary(self, data: dict) -> None:
        if self.run:
            self.run.summary.update(data)

    def image(self, key: str, path: Path) -> None:
        if self.run:
            import wandb
            self.run.log({key: wandb.Image(str(path))})

    def model_artifact(self, path: Path, name: str, metadata: dict) -> None:
        """Versioned copy of the checkpoint in W&B, so teammates can download the exact weights."""
        if self.run:
            import wandb
            art = wandb.Artifact(name, type="model", metadata=metadata)
            art.add_file(str(path))
            self.run.log_artifact(art)

    def finish(self) -> None:
        if self.run:
            self.run.finish()
