#from transformers import FlaxAutoModelForSeq2SeqLM
#from transformers import AutoTokenizer
# https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch
import pytorch_lightning as pl

generation_kwargs = {
    "max_length": 512,
    "min_length": 64,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
    "top_k": 60,
    "top_p": 0.95
}
# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html
class ChefT5(pl.LightningModule):

    def __init__(self, size_train, lr=5e-5, num_train_epochs=5, warmup_steps=1000):
        #super(ChefT5, self).__init__()
        super().__init__()
        self.size_train = size_train
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.save_hyperparameters() 
        # https://github.com/Lightning-AI/lightning/issues/3981

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss  # outputs.logits possible as well! # Seq2SeqSequenceClassifierOutput
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        #tokenizer = T5Tokenizer.from_pretrained("t5-small")
        #directions = batch["labels"]
        #print(directions.shape)
        #raise SystemExit
        #print(tokenizer.decode())
        #raise SystemExit
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)
        #TODO: https://github.com/kelvin-jose/T5-Transformer-Lightning/blob/master/model.py
        # NO TEXT GENERATION
        #print(batch["labels"])
        ## now preds
        #outputs = self(**batch)
        #logits = outputs.logits
        #preds = torch.argmax(logits, axis=1)
        #print(preds)
        #raise SystemExit
        return loss

    def test_step(self, batch, batch_idx):
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        loss = self.common_step(batch, batch_idx)
        inputs = batch["input_ids"]
        attention = batch["attention_mask"]
        outputs = self.model.generate(input_ids=inputs, attention_mask=attention, **generation_kwargs)  # RELOAD MODEL
        #model = T5ForConditionalGeneration.from_pretrained(save_directory)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print('*'*8)
        print(generated)
        print('*'*8)
        masks = batch["labels"][0] != -100
        clean_labels = torch.masked_select(batch["labels"][0], masks)
        print(tokenizer.decode(clean_labels, skip_special_tokens=True))
        print('*'*8)
        masks = batch["input_ids"][0] != -100
        clean_ingredients = torch.masked_select(batch["input_ids"][0], masks)
        print(tokenizer.decode(clean_ingredients, skip_special_tokens=True))
        raise SystemExit
        return loss
    
    #def generate_recipe(self, batch):
        #pass

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        # lr scheduler
        num_train_optimization_steps = self.hparams.num_train_epochs * self.size_train
        lr_scheduler = {"scheduler": get_linear_schedule_with_warmup(optimizer,
                         num_warmup_steps=self.hparams.warmup_steps, num_training_steps=num_train_optimization_steps),
                        "name": "learning_rate", "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}