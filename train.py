import wandb
import pandas as pd
import torch.cuda as cuda
from datasets import load_dataset
from pytorch_lightning import Trainer
from transformers import T5ForConditionalGeneration
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from custom_T5 import ChefT5
from dataloader_preprocessing import RecipeTXTData
from recipe_generation import generate_recipes

# https://open.gitcode.host/wandb-docs/library/init.html
# includes offline doc
#wandb.login()  # need acc?
#wandb.init(project="test-project", entity="multimodal_projs")

#wandb.config = {
#  "learning_rate": 0.001,
#  "epochs": 100,
#  "batch_size": 128
#}
#
#wandb.log({"loss": loss})
#

data_reader = RecipeTXTData()  # should have path as global var in env?
# https://www.youtube.com/watch?v=r6XY80Z9eSA
#data_reader.prepare_data()
data_reader.setup()

#dataset = load_dataset("Zappandy/recipe_nlg")
#print(dataset)

test_loader = data_reader.test_dataloader()
val_loader = data_reader.val_dataloader()
train_loader = data_reader.train_dataloader()
size_train = len(train_loader)


#generate_recipes(data_reader.test_data)
model = ChefT5(size_train=size_train)

## Optional
#wandb.watch(model)

MIN_EPOCHS = 1
MAX_EPOCHS = 2
use_gpu = False
if cuda.is_available() and use_gpu:
    trainer = Trainer(accelerator='gpu', devices=1, min_epochs=MIN_EPOCHS, max_epochs=MAX_EPOCHS, log_every_n_steps=2)  # default n_steps == 50
else:
    trainer = Trainer(min_epochs=MIN_EPOCHS, max_epochs=MAX_EPOCHS, log_every_n_steps=2)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(dataloaders=train_loader, ckpt_path='best')  # withoutcheckpoint previous fit is used? Better store/define model to load with test
#wandb.finish()

save_directory = "./stored_models/torch_chef/"
model.model.save_pretrained(save_directory)


#TODO: WHEN WE ARE DONE UNCOMMENT LINES IN prepare_data in lightning data module
