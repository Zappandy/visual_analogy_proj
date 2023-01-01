#from transformers import FlaxAutoModelForSeq2SeqLM
#from transformers import AutoTokenizer
# https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
from transformers import T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl

class ChefT5(pl.LightningModule):

    def __init__(self, size_train, lr=5e-5, num_train_epochs=5, warmup_steps=1000):
        #super(ChefT5, self).__init__()
        super().__init__()
        self.size_train = size_train
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.save_hyperparameters()  # this uses the lr and warmup steps. NOT USING EPOCHS!
        # https://github.com/Lightning-AI/lightning/issues/3981

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        # lr scheduler
        num_train_optimization_steps = self.hparams.num_train_epochs * self.size_train
        lr_scheduler = {"scheduler": get_linear_schedule_with_warmup(optimizer,
                         num_warmup_steps=self.hparams.warmup_steps, num_training_steps=num_train_optimization_steps),
                        "name": "learning_rate", "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    #def train_dataloader(self):
    #    return self.train_dataloader

    #def val_dataloader(self):
    #    return self.val_dataloader

    #def test_dataloader(self):
    #    return self.test_dataloader

#tokenizer = T5Tokenizer.from_pretrained("t5-small")
#model = T5ForConditionalGeneration.from_pretrained("t5-small")

#prefix = "items: "
#
#generation_kwargs = {"max_length": 256, "min_length": 32,
#                     "no_repeat_ngram_size": 3, "do_sample": True,
#                     "top_k": 60, "top_p": 0.95}  # 512, and min 64
#
#special_tokens = tokenizer.all_special_tokens
#tokens_map = {"<sep>": "--", "<section>": "\n"}
#
## pass list of items, tokens. This generates text
#def skip_special_tokens(text, special_tokens):
#    j
#
