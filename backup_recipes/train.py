import wandb
import pandas as pd
import torch.cuda as cuda
from datasets import load_dataset
from pytorch_lightning import Trainer
from transformers import T5ForConditionalGeneration
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from custom_T5 import ChefT5
from data_exploration import RecipeTXTData

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
data_reader.prepare_data()
# https://www.youtube.com/watch?v=r6XY80Z9eSA
data_reader.setup()
test_loader = data_reader.test_dataloader()
val_loader = data_reader.val_dataloader()
train_loader = data_reader.train_dataloader()
size_train = len(train_loader)


from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from demo.utils import generation_kwargs
import torch

model = T5ForConditionalGeneration.from_pretrained('.')
tokenizer = T5Tokenizer.from_pretrained("t5-small")

test_example = next(iter(data_reader.test_data))

inputs = test_example["input_ids"]
labels = test_example["labels"]
attention = test_example["attention_mask"]
outputs = model.generate(input_ids=torch.unsqueeze(inputs, 0), attention_mask=torch.unsqueeze(attention, 0), **generation_kwargs)  # RELOAD MODEL

inp_masks = inputs != -100
label_masks = labels != -100
clean_inputs = torch.masked_select(inputs, inp_masks)
clean_labels = torch.masked_select(labels, label_masks)
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print()
print('*'*8 + 'INPUTS')
print(tokenizer.decode(clean_inputs, skip_special_tokens=True))
print('*'*8 + 'GENERATED')
print(generated)
print('*'*8 + 'LABELS')
print(tokenizer.decode(clean_labels, skip_special_tokens=True))
raise SystemExit
#----- TESTING MODEL


model = T5ForConditionalGeneration.from_pretrained('.')

test_example = dataset['test'][2]
print("Code:", test_example['code'])

# prepare for the model
input_ids = tokenizer(test_example['code'], return_tensors='pt').input_ids
# generate
outputs = model.generate(input_ids)
print("Generated docstring:", tokenizer.decode(outputs[0], skip_special_tokens=True))

print("Ground truth:", test_example['docstring'])
#-------------

raise SystemExit
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


#TODO: WHEN WE ARE DONE UNCOMMENT LINES IN prepare_data in lightning data module