import wandb
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from custom_T5 import ChefT5
from data_exploration import RecipeTXTData

#wandb.login()  # need acc?


data_reader = RecipeTXTData()  # should have path as global var in env?
data_reader.prepare_data()
data_reader.setup()
test = data_reader.test_dataloader()
for i in test:
    print(i.keys())
    break
    print(i.shape, l.shape)

#TODO: complete preprocessing with pl's data module. It will yield data loaders to feed to model
#dataset = data_reader.split_data()
#chef_t5 = ChefT5(dataset)
#print(chef_t5.test_dataloader.columns)  # title, ingredients, directions, NER
