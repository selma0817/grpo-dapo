"""Train the configured GRPO policy."""

from grpo_dapo.config import load_config
from grpo_dapo.trainer import train


if __name__ == "__main__":
    train(load_config())
