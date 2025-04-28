# train_dpo.py
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import DPOConfig, DPOTrainer


def main():
    # Load dataset
    train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:1024]")

    # Load model
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    # model_id = "unsloth/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Train model
    training_args = DPOConfig(
        output_dir=f"{model_id}-codeforces-SFT",
        logging_steps=10,
        bf16=True,
        use_liger_kernel=False,
        gradient_checkpointing=False,
        max_length=500,
        max_prompt_length=128,
        max_completion_length=128,
        dataset_num_proc=32,
        num_train_epochs=1,
        eval_strategy="steps",
        do_eval=True,
        optim="paged_adamw_8bit",
        eval_steps=10,
        max_steps=10,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        args=training_args,
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
