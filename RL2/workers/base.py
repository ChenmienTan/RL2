from omegaconf import DictConfig
from transformers import AutoTokenizer


class Worker:

    def __init__(self, config: DictConfig, train: bool):

        self.config = config
        self.train = train

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )