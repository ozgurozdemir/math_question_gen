from transformers import BertGenerationEncoder, BertGenerationDecoder
from transformers import AutoTokenizer, GPT2Model, GPT2LMHeadModel, AutoConfig, AutoModelForTokenClassification
from transformers import EncoderDecoderModel
from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch

class Seq2SeqModel(torch.nn.Module):
    def __init__(self, tokenizer, encoder="bert", decoder="gpt"):
        if encoder == "bert":
            encoder_model = BertGenerationEncoder.from_pretrained("AnReu/math_albert",
                                                            bos_token_id=tokenizer.bos_token,
                                                            eos_token_id=tokenizer.eos_token)

        else: 
            config = AutoConfig.from_pretrained("gpt2", vocab_size=len(tokenizer), n_ctx=128, 
                                                bos_token_id=tokenizer.bos_token_id,
                                                eos_token_id=tokenizer.eos_token_id)
            encoder_model = GPT2Model(config)

        if decoder == "bert":
            decoder_decoder = BertGenerationDecoder.from_pretrained("AnReu/math_albert",
                                                            add_cross_attention=True,
                                                            is_decoder=True,
                                                            bos_token_id=tokenizer.bos_token,
                                                            eos_token_id=tokenizer.eos_token)
            
        else:
            config = AutoConfig.from_pretrained("gpt2", add_cross_attention=True, vocab_size=len(tokenizer), n_ctx=128,
                                                bos_token_id=train_dataset.tokenizer.bos_token_id,
                                                eos_token_id=train_dataset.tokenizer.eos_token_id)
            decoder_model = GPT2LMHeadModel(config)

        encoder_model.resize_token_embeddings(len(tokenizer))
        decoder_model.resize_token_embeddings(len(tokenizer))

        self.model = EncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        
        self.model.config.decoder_start_token_id = tokenizer.cls_token_id
        self.model.config.eos_token_id = tokenizer.sep_token_id
        self.model.config.pad_token_id = tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.encoder.vocab_size

        model_size = sum(t.numel() for t in self.model.parameters())
        print(f"Seq2Seq model with enc: {encoder} dec: {decoder}, total size: {model_size/1000**2:.1f}M parameters")

    
    def prepare_trainer_object(self, train_dataset, valid_dataset):
        data_collator = DataCollatorForSeq2Seq(self.tokenizer)

        args = Seq2SeqTrainingArguments(
            predict_with_generate=False,
            output_dir="./seq2seq",
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

        trainer = Seq2SeqTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset
        )

        return trainer