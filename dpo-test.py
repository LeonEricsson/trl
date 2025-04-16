
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig
from utils import *

def main():
    ######## MODEL #############
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    ############ DATASET ############
    # source: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_(7B)-DPO.ipynb#scrollTo=AqkY_wHdKyOl
    raw_datasets = get_datasets(
        {"HuggingFaceH4/ultrafeedback_binarized" : 0.005}, # 0.5% sampled
        splits = ["train_prefs", "test_prefs"],
    )
    column_names = list(raw_datasets["train"].features)

    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs = {"tokenizer": tokenizer, "task": "dpo"},
        num_proc = 12,
        remove_columns = column_names,
        desc = "Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    config =  DPOConfig(
        report_to = "none",
        output_dir="outputs",
        eval_strategy="steps",
        do_eval=True,
        optim="paged_adamw_8bit",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=1,
        log_level="debug",
        save_strategy="steps",
        save_steps=200,
        logging_steps=25,
        learning_rate=5e-6,
        bf16 = True,
        beta = 0.1,
        eval_steps=10,
        #num_train_epochs=1,
        max_steps=10,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        model_adapter_name="DPO",
        ref_adapter_name="reference",
        max_length = 256,
        max_prompt_length = 128)
    
    dpo_trainer = DPOTrainer(
        model = model,
        ref_model = None,
        args = config,
        train_dataset = raw_datasets["train"],
        processing_class = tokenizer,
        eval_dataset=raw_datasets['test']
    )

    dpo_trainer.train()


if __name__ == "__main__":
    main()