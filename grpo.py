from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import GRPOConfig, GRPOTrainer


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": "test",
            "answer": extract_hash_answer(x["answer"]),
        }
    )  # type: ignore
    return data  # type: ignore


# Reward functions
def correctness_reward_func(completions, **kwargs):
    return [1]


def main():
    dataset = get_gsm8k_questions()

    # wandb.init(
    #     project="TRL",
    #     notes="Debug runs for TRL",
    #     mode="online",
    # )

    # model_id = "unsloth/Llama-3.2-1B-Instruct"
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    training_args = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,  # Increase to 4 for smoother training
        num_generations=2,  # Decrease if out of memory
        max_prompt_length=50,
        max_completion_length=25,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps=250,
        save_steps=250,
        max_grad_norm=1.0,
        report_to="none",  # Can use Weights & Biases
        output_dir="outputs",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=correctness_reward_func,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    # wandb.finish()


if __name__ == "__main__":
    main()
