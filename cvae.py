import transformers
import torch
import copy
from dataclasses import dataclass

from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoModelForTokenClassification
from transformers import AutoConfig, AutoModel
from transformers import RobertaForCausalLM

from transformers.modeling_outputs import BaseModelOutput
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments

from peft import LoraConfig, get_peft_model
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

@dataclass
class VAEModelOutput(BaseModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    loss_kl: torch.FloatTensor = None
    loss_r: torch.FloatTensor = None
    loss_cls: torch.FloatTensor = None


class VAE_Bert_Model(torch.nn.Module):
    def __init__(self, vocab_size, maxlen, num_subj, latent_dim, device, 
                 use_gpt=False, pretrained_model=None, encoder=None, decoder=None):
        super(VAE_Bert_Model, self).__init__()

        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.num_subj = num_subj
        self.latent_dim = latent_dim
        self.device = device
        self.use_gpt = use_gpt

        # encoder and decoder networks
        if self.use_gpt:
            self.encoder = encoder
            self.decoder = decoder
        else:
            self.encoder = pretrained_model
            self.decoder = copy.deepcopy(pretrained_model)

        self.decoder.classifier = torch.nn.Sequential(*[torch.nn.Linear(self.latent_dim, self.vocab_size),
                                                        torch.nn.Softmax(dim=1)])

        # posterior and prior networks
        posterior_dim = (self.maxlen-1) * self.latent_dim

        self.posterior_mu  = torch.nn.Sequential(*[torch.nn.Linear(posterior_dim, self.latent_dim),
                                                   torch.nn.Sigmoid()])
        self.posterior_var = torch.nn.Sequential(*[torch.nn.Linear(posterior_dim, self.latent_dim),
                                                   torch.nn.Sigmoid()])

        self.prior_mu  = torch.nn.Sequential(*[torch.nn.Linear(self.latent_dim, self.latent_dim),
                                               torch.nn.Sigmoid()])
        self.prior_var = torch.nn.Sequential(*[torch.nn.Linear(self.latent_dim, self.latent_dim),
                                               torch.nn.Sigmoid()])

        # subject prediction network
        self.subj_classifier = torch.nn.Sequential(*(torch.nn.Linear(self.latent_dim, self.num_subj),
                                                     torch.nn.Softmax(dim=1)))


    def reparameterize(self, inp, network):
        mean = network[0](inp)
        logv = network[1](inp)
        std  = torch.exp(0.5 * logv)

        z = torch.randn(inp.shape[0], self.latent_dim).to(self.device)
        z = z * std + mean

        return z


    def encode(self, inp, att):
        if self.use_gpt:
            enc_hidden = self.encoder.encoder(inp, att).last_hidden_state
        else:
            enc_hidden = self.encoder.roberta(inp, att).last_hidden_state

        post_hidden = enc_hidden[:, 1:, :].view(enc_hidden.shape[0], -1)
        post_z  = self.reparameterize(post_hidden, [self.posterior_mu, self.posterior_var])

        prior_hidden = enc_hidden[:, 0, :] # CLS token
        prior_z = self.reparameterize(prior_hidden, [self.prior_mu, self.prior_var])

        return post_z, prior_z


    def decode(self, inp, att, post_z):
        if self.use_gpt:
            dec_inp = self.decoder.transformer.wte(inp)
            dec_inp = torch.cat((post_z.unsqueeze(1), dec_inp[:, 1:]), dim=1) # adding z to the decoder input
    
            for h in self.decoder.transformer.h:
                out = h(out[0])
    
            out = self.decoder.transformer.ln_f(out[0])
            out = self.decoder.lm_head(out)   
            
        else:
            dec_inp = self.decoder.roberta.embeddings(inp)
            dec_inp = torch.cat((post_z.unsqueeze(1), dec_inp[:, 1:]), dim=1) # adding z to the decoder input

            out = self.decoder.roberta.encoder(dec_inp)[0]
            out = self.decoder.classifier(out)

        return out


    def classify_subject(self, post_z, prior_z):
        post_subj  = self.subj_classifier(post_z)
        prior_subj = self.subj_classifier(prior_z)

        return post_subj, prior_subj


    def forward(self, input_ids, attention_mask, labels=None):
        inp = input_ids
        att = attention_mask
        tar = labels

        post_z, prior_z = self.encode(inp, att)
        out = self.decode(inp, att, post_z)

        post_subj, prior_subj = self.classify_subject(post_z, prior_z)

        if labels is not None:
            loss, loss_r, loss_cls, loss_kl = self.loss_func(inp, out, tar, post_z, prior_z, post_subj, prior_subj)
            return VAEModelOutput(loss=loss, logits=out, loss_kl=loss_kl, loss_r=loss_r, loss_cls=loss_cls)
        else:
            return VAEModelOutput(logits=out)


    def loss_subj(self, post_subj, prior_subj, tar):
        loss_post  = torch.nn.functional.cross_entropy(post_subj, tar)
        loss_prior = torch.nn.functional.cross_entropy(prior_subj, tar)

        return loss_post + loss_prior


    def loss_kl(self, post_z, prior_z):
        post_z = torch.nn.functional.log_softmax(post_z, dim=1)
        prior_z = torch.nn.functional.softmax(prior_z, dim=1)

        return torch.nn.functional.kl_div(post_z, prior_z)


    def loss_reconstruct(self, out, tar):
        out = out.view(-1, self.vocab_size)
        tar = tar.view(-1)

        return torch.nn.functional.cross_entropy(out, tar)

    def loss_func(self, inp, out, tar, post_z, prior_z, post_subj, prior_subj):
        loss_r = self.loss_reconstruct(out, inp)
        loss_kl = self.loss_kl(post_z, prior_z)
        loss_cls = self.loss_subj(post_subj, prior_subj, tar)

        loss = loss_r + loss_cls - loss_kl
        return loss, loss_r, loss_cls, loss_kl



class VAEConfig(PretrainedConfig):
    model_type = "vae"

    def __init__(self, premodel = "witiko/mathberta", 
                 vocab_size = 30532,
                 maxlen = 128,
                 num_subj = 7,
                 latent_dim = 768,
                 device = "cuda",
                 use_gpt = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.premodel = premodel
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.num_subj = num_subj
        self.latent_dim = latent_dim
        self.device = device
        self.use_gpt = use_gpt



class VAE_Model(PreTrainedModel):
    config_class = VAEConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        if config.use_gpt:
            enc_config = AutoConfig.from_pretrained(
                "gpt2",
                vocab_size=config.vocab_size,
                n_ctx=config.maxlen,
                bos_token_id=2,
                eos_token_id=3)
            encoder = GPT2Model(enc_config)

            dec_config = AutoConfig.from_pretrained(
                "gpt2",
                add_cross_attention=True,
                vocab_size=config.vocab_size,
                n_ctx=config.maxlen,
                bos_token_id=2,
                eos_token_id=3)
            decoder = GPT2LMHeadModel(dec_config)

            encoder.resize_token_embeddings(config.vocab_size)
            decoder.resize_token_embeddings(config.vocab_size))

            self.model = VAE_Bert_Model(config.vocab_size,
                                        config.maxlen,
                                        config.num_subj,
                                        config.latent_dim,
                                        config.device,
                                        encoder=encoder, 
                                        decoder=decoder,
                                        use_gpt=True)
        else:
            premodel = RobertaForCausalLM.from_pretrained(config.premodel)
            premodel.resize_token_embeddings(config.vocab_size)
            premodel.classifier = torch.nn.Sequential(*[torch.nn.Linear(config.latent_dim, config.vocab_size),
                                                        torch.nn.Softmax(dim=1)])
    
            self.model = VAE_Bert_Model(config.vocab_size,
                                        config.maxlen,
                                        config.num_subj,
                                        config.latent_dim,
                                        config.device,
                                        pretrained_model = premodel,
                                        use_gpt=False)

        model_size = sum(t.numel() for t in self.model.parameters())
        print(f"CVAE model with use_gpt: {config.use_gpt}, total size: {model_size/1000**2:.1f}M parameters")
        

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask, labels)

    def generate(self, input_ids, **kwargs):
        if self.config.use_gpt:
            inp = self.model.encoder(input_ids)[0]
        else:
            inp = self.model.encoder.roberta(input_ids)[0]
            
        cls = inp[:, 0, :] # CLS token
        z   = self.model.reparameterize(cls, [self.model.prior_mu, self.model.prior_var])
        
        with torch.no_grad():
            inputs_embeds = torch.normal(mean=0., std=1., size=(input_ids.shape[0], 1, self.model.latent_dim), device="cuda")
            inputs_embeds = torch.cat((z.unsqueeze(1), inputs_embeds[:, 1:, :]), dim=1) # adding z to the decoder input

        if self.use_gpt:
            return self.model.decoder.generate(inputs_embeds=inputs_embeds, **kwargs)
            
        else:
            inputs_embeds = self.model.decoder.classifier(inputs_embeds)
            inputs_embeds = inputs_embeds.argmax(dim=-1)
            inputs_embeds = torch.cat((input_ids[:, 0], inputs_embeds[:, 0]), dim=0) # adding z to the decoder input
            
            return self.model.decoder.generate(input_ids, **kwargs)
        
    def prepare_lora_model(self):
        config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["posterior_mu.0", "posterior_var.0", "prior_mu.0", "prior_var.0", "subj_classifier.0"],
            lora_dropout=0.1,
            bias="lora_only",
            modules_to_save=["decoder.lm_head"]
        )

        self.model = get_peft_model(self.model, config)
        lora_model.print_trainable_parameters()

    
    def prepare_trainer_object(self, train_dataset, valid_dataset, use_lora=False):
        train_dataset.tokenizer.pad_token = train_dataset.tokenizer.eos_token
        data_collator = DataCollatorWithPadding(train_dataset.tokenizer)

        if use_lora:
            self.prepare_lora_model()
        
        data_collator = 
        args = TrainingArguments(
            output_dir="./cvae",
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
        
        return trainer

