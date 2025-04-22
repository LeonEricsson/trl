from datasets import load_dataset
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer


def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    # model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    # model_id = "unsloth/Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # dataset = load_dataset("trl-lib/tldr", split="train[:2048]")
    # eval_dataset = load_dataset("trl-lib/tldr", split="validation[:256]")
    dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train[:2048]")
    eval_dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="test[:256]")

    # Dummy reward function: count the number of unique characters in the completions
    def reward_num_unique_chars(completions, **kwargs):
        return [len(set(c)) for c in completions]

    config = GRPOConfig(
        output_dir=f"{model_id}-GRPO",
        logging_steps=10,
        bf16=True,
        # use_liger_kernel=True,
        max_completion_length=128,
        gradient_checkpointing=True,
        use_vllm=False,
        num_iterations=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        do_eval=True,
        optim="paged_adamw_8bit",
        eval_steps=10,
        max_steps=10,
    )

    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs=reward_num_unique_chars,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=config,
    )

    trainer.train()


if __name__ == "__main__":
    main()
