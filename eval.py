from transformers import pipeline
import code_bert_score
from tqdm import tqdm
import numpy as np
import torch
import json

def prepare_pipeline(model, tokenizer):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

    return pipe


def generate_examples(pipe, num_samples, save_dir="./results", save_name="generations"):
    generations = {
        'Algebra': ["[ALGEBRA]"]*num_samples["Algebra"],
        'Counting & Probability': ["[COUNTPROB]"]*num_samples["Counting & Probability"],
        'Geometry': ["[GEO]"]*num_samples["Geometry"],
        'Intermediate Algebra': ["[INTERALGEBRA]"]*num_samples["Intermediate Algebra"],
        'Number Theory': ["[NUMTHEORY]"]*num_samples["Number Theory"],
        'Prealgebra': ["[PREALGEBRA]"]*num_samples["Prealgebra"],
        'Precalculus': ["[PRECALCULUS]"]*num_samples["Precalculus"]
    }

    for g in tqdm(generations):
        generations[g] = pipe(generations[g],
                              num_beams=5,
                              num_beam_groups=5,
                              pad_token_id=3,
                              max_length=100,
                              diversity_penalty=1.0)

    for gen in generations:
        generations[gen] = [g[0]["generated_text"] for g in generations[gen]]

    with open(f"{save_dir}/{model}.json", "w") as file:
        json.dump(generations, file)

    return generations


def prepare_eval_set(test_dataset):
    print(">> Preparing the eval set..")
    examples = {
        'Algebra': [],
        'Counting & Probability': [],
        'Geometry': [],
        'Intermediate Algebra': [],
        'Number Theory': [],
        'Prealgebra': [],
        'Precalculus': []
    }
    
    for i in tqdm(range(len(test_dataset))):
        cls = test_dataset.subj_to_token[test_dataset.id_to_subj[test_dataset[i]["labels"]]]
        tok = test_dataset.tokenizer.decode(test_dataset[i]["input_ids"], skip_special_tokens=True).replace(f"{cls} ", "")
        examples[test_dataset.id_to_subj[test_dataset[i]["labels"]]].append(tok)


def run_evaluation(file_path, test_dataset):
    with open(file_path, "r") as file:
        generations = json.load(file)
        
    examples = prepare_eval_set(test_dataset)
    
    code_bert_scores = {}

    for gen in generations:
        gs = [d.replace(f"{test_dataset.subj_to_token[gen]} ", "") for d in generations[gen]]
        ex = [d.replace(f"{test_dataset.subj_to_token[gen]} ", "") for d in examples[gen]]
    
        pred_results = code_bert_score.score(cands=gs, refs=ex, model_type="AnReu/math_albert")
        code_bert_scores[gen] = {"p":  pred_results[0].mean().unsqueeze(dim=0).numpy().astype(np.float64)[0],
                                  "r":  pred_results[1].mean().unsqueeze(dim=0).numpy().astype(np.float64)[0],
                                  "f1": pred_results[2].mean().unsqueeze(dim=0).numpy().astype(np.float64)[0],
                                  "f3": pred_results[3].mean().unsqueeze(dim=0).numpy().astype(np.float64)[0]}

    average = np.mean([code_bert_scores[gen]["f1"] for gen in code_bert_scores])

    print("Average", average)
    for gen in code_bert_scores:
        print(gen, code_bert_scores[gen]["f1"])
