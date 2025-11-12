class LlamaFineTuningConfig:
    """Configuration class for Llama fine-tuning experiments.

    Updated with chat template optimizations based on successful Qwen3 implementation.
    """

    def __init__(
        self,
        model_name: str,
        train_file: str = "data/train.jsonl",
        output_dir: str = "./results_clean",
        batch_size: int = 4,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 5e-5,
        num_epochs: int = 1,
        max_length: int = 512,
        lora_r: int = 24,
        lora_alpha: int = 48,
        lora_dropout: float = 0.1,
        gradient_checkpointing: bool = True,
        lr_scheduler_type: str = "cosine",
    ):
        self.model_name = model_name
        self.train_file = train_file
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.lr_scheduler_type = lr_scheduler_type

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size."""
        return self.batch_size * self.gradient_accumulation_steps

    def print_config(self):
        """Print current configuration."""
        print(f"✅ Configuration set")
        print(f"Model: {self.model_name}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Effective batch size: {self.effective_batch_size}")
        print(
            f"LoRA: r={self.lora_r}, alpha={self.lora_alpha}, dropout={self.lora_dropout}"
        )
        print(f"Scheduler: {self.lr_scheduler_type}")
