from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch

class DecoderModel():
    def __init__(self):       
       config = AutoConfig.from_pretrained(
            "gpt2",
            vocab_size=len(train_dataset.tokenizer),
            n_ctx=128,
            bos_token_id=2,
            eos_token_id=3,
        )

        self.model = GPT2LMHeadModel(config)
        
        model_size = sum(t.numel() for t in model.parameters())
        print(f"GPT-2 model size: {model_size/1000**2:.1f}M parameters")


    def prepare_trainer_object(self, train_dataset, valid_dataset):
        train_dataset.tokenizer.pad_token = train_dataset.tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(train_dataset.tokenizer, mlm=False)

        args = TrainingArguments(
            output_dir="./gpt2",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
        
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            logging_steps=100,
            gradient_accumulation_steps=8,
            num_train_epochs=20,
            weight_decay=0.1,
            warmup_steps=100,
            lr_scheduler_type="cosine",
            learning_rate=5e-4,
            save_steps=100,
            fp16=True,
            push_to_hub=False,

            save_total_limit = 1,
            load_best_model_at_end=True
        )
        
        trainer = Trainer(
            model=self.model,
            tokenizer=train_dataset.tokenizer,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
        )