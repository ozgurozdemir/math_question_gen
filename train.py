from transformers import DataCollatorWithPadding
from decoder_based import DecoderModel
from dataset import MathDataset
from seq2seq import Seq2SeqModel
from cvae import *
from eval import *
import torch

if __name__ == "__main__":

    ## dataset preparation
    train_dataset = MathDataset(data_path="/MATH/train", tokenizer="AnReu/math_albert", maxlen=128, seq2seq=False)
    valid_dataset = MathDataset(data_path="./MATH/test",  tokenizer="AnReu/math_albert", maxlen=128, seq2seq=False)

    train_dataset.tokenizer.pad_token = train_dataset.tokenizer.eos_token
    data_collator = DataCollatorWithPadding(train_dataset.tokenizer)

    ## model selection
    # decoder model
    # model = DecoderModel()
    
    # seq2seq model
    # model = Seq2SeqModel(tokenizer, encoder="bert", decoder="gpt")

    # cvae model
    AutoConfig.register("vae", VAEConfig)
    AutoModel.register(VAEConfig, VAE_Model)
    
    config = VAEConfig(
        vocab_size = len(train_dataset.tokenizer),
        latent_dim = 768,
        maxlen = 128,
        num_subj = 7,
        device = "cuda" if torch.cuda.is_available() else "cpu"),
        use_gpt=True,
        # premodel = "witiko/mathberta"
    )
    
    model = VAE_Model(config)

    ## training 
    trainer = model.prepare_trainer_object(train_dataset, valid_dataset)

    trainer.train()
    train.save_model("./cvae")

    ## evaluation
    num_samples = {
        'Algebra': 1187,
        'Counting & Probability': 474,
        'Geometry': 479,
        'Intermediate Algebra': 903,
        'Number Theory': 540,
        'Prealgebra': 871,
        'Precalculus': 546
    }

    pipeline= prepare_pipeline(model, train_dataset.tokenizer)
    generations = generate_examples(pipeline, num_samples, save_dir="./results", save_name="cvae_gpt")

    
    run_evaluation("./results/cvae_gpt.json", valid_dataset)