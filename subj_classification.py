from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import confusion_matrix, classification_report
from transformers import DataCollatorWithPadding
from transformers import AutoConfig
from transformers import pipeline
from dataset import MathDataset
from tqdm import tqdm
from eval import *
import numpy as np
import evaluate
import torch



metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return metric.compute(predictions=predictions, references=labels, average="macro")


def prepare_trainer(model="ozgurozdemir/math-gen-gpt2", train_dataset, valid_dataset):
    data_collator = DataCollatorWithPadding(tokenizer=train_dataset.tokenizer)
    
    config = AutoConfig.from_pretrained(
        model,                                 
        num_labels=7, 
        id2label=id2label, 
        label2id=label2id)
    
    config.vocab_size = len(train_dataset.tokenizer)
    config.pad_token_id = train_dataset.tokenizer.pad_token_id

    model = AutoModelForSequenceClassification.from_config(config)

    training_args = TrainingArguments(
        output_dir=".bert-cls",
        learning_rate=5e-4,
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
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=train_dataset.tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


def predict(pipeline, test_dataset):
    preds = []
    gts = []

    for i in tqdm(range(len(test_dataset))):
        inp = test_dataset.tokenizer.decode(test_dataset[i]["input_ids"][1:], skip_special_tokens=True)
        pred = pipeline(inp)[0]["label"]

        preds.append(test_dataset.subj_to_id[pred])
        gts.append(test_dataset[i]["labels"])

    return gts, preds


def display_results(gts, preds):
    print(confusion_matrix(gts, preds))
    print("-"*30)
    print(classification_report(gts, preds))


if __name__ == "__main__":

    ## dataset preparation
    train_dataset = MathDataset(data_path="/MATH/train", tokenizer="AnReu/math_albert", maxlen=128, replace_cls=False)
    valid_dataset = MathDataset(data_path="./MATH/test",  tokenizer="AnReu/math_albert", maxlen=128, replace_cls=False)
   
    train_dataset.tokenizer.pad_token = train_dataset.tokenizer.eos_token
    data_collator = DataCollatorWithPadding(train_dataset.tokenizer)

    ## model selection
    model = "ozgurozdemir/math-gen-gpt2"
        
    ## training 
    trainer = prepare_trainer_object(model=model, train_dataset, valid_dataset)

    trainer.train()
    train.save_model("./subj_cls")

    ## evaluation
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe = pipeline("text-classification", model="./subj_cls", device=device)

    gts, preds = predict(pipeline, valid_dataset)
    display_results(gts, preds)