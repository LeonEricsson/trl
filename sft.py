# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Train Gemma-3 on the Codeforces COTS dataset.

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml examples/scripts/sft_gemma3.py
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import SFTConfig, SFTTrainer


def main():
    # Load dataset
    train_dataset = load_dataset("open-r1/codeforces-cots", split="train")
    train_dataset = train_dataset.remove_columns("messages")

    # Load model
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    # model_id = "unsloth/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # print(tokenizer.padding_side)
    def formatting_function(examples):
        return f"{examples['prompt']} \n {examples['generation']}"

    # Train model
    training_args = SFTConfig(
        output_dir=f"{model_id}-codeforces-SFT",
        logging_steps=10,
        bf16=True,
        use_liger_kernel=True,
        gradient_checkpointing=False,
        max_length=500,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        dataset_num_proc=32,
        num_train_epochs=1,
        eval_strategy="steps",
        do_eval=True,
        optim="paged_adamw_8bit",
        eval_steps=10,
        max_steps=10,
        report_to="none",
    )
    trainer = SFTTrainer(
        args=training_args,
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        formatting_func=formatting_function,
    )
    trainer.train()


if __name__ == "__main__":
    main()
