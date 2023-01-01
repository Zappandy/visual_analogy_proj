import wandb
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from custom_T5 import ChefT5
from data_exploration import RecipeTXTData

# https://open.gitcode.host/wandb-docs/library/init.html
# includes offline doc
#wandb.login()  # need acc?
#
#wandb.init(project="test-project", entity="multimodal_projs")

#wandb.config = {
#  "learning_rate": 0.001,
#  "epochs": 100,
#  "batch_size": 128
#}
#
#wandb.log({"loss": loss})
#
## Optional
#wandb.watch(model)

data_reader = RecipeTXTData()  # should have path as global var in env?
data_reader.prepare_data()
data_reader.setup()
test_loader = data_reader.test_dataloader()
val_loader = data_reader.val_dataloader()
train_loader = data_reader.train_dataloader()
size_train = len(train_loader)
model = ChefT5(size_train=size_train)  # if loaders are loaded in batches in data module, we must change this
trainer = Trainer(max_epochs=3)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
#trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, datamodule=data_reader)

#TODO: complete preprocessing with pl's data module. It will yield data loaders to feed to model
#dataset = data_reader.split_data()
#chef_t5 = ChefT5(dataset)
#print(chef_t5.test_dataloader.columns)  # title, ingredients, directions, NER
