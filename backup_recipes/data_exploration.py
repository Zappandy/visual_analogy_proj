#from transformers import FlaxAutoModelForSeq2SeqLM
#from transformers import AutoTokenizer
import pandas as pd
import ast
#from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningDataModule
from datasets import Dataset, load_from_disk
from transformers import T5Tokenizer#Fast


SEED = 42
path = "data/dataset/full_dataset.csv"
test_path = "./example_recipes.csv"
max_input_length = 128

class RecipeTXTData(LightningDataModule):

    #def __init__(self, data_dir: str=path):
    def __init__(self, data_csv: str=test_path, data_dir: str="testing_stuff"):
        super().__init__()
        self.data_csv = data_csv
        self.data_dir = data_dir
        self.batch_size = 16
        self.sentinel_tkn = "<extra_id_99>"


    def prepare_data(self):

        # https://github.com/huggingface/transformers/issues/16986
        df = pd.read_csv(self.data_csv)
        # DO NOT DELETE THESE 3 LINES ARE FOR FULL DATASET
        #df = pd.read_csv(self.data_csv, index_col=0)  # to clean weird idx
        #df.drop(["source", "link"], axis=1, inplace=True)  # only use with real_file
        #self.df.reset_index(drop=True, inplace=True)
        headers = ["ingredients", "NER", "directions"]  # title
        df = self.preprocess_lists(df, headers)

        # tokenize...
        raw_dataset = Dataset.from_pandas(df)
        raw_dataset = raw_dataset.map(self.preprocess_tokenize, batched=True)
        # according to doc, it's better to store in local
        #self.dataset.to_csv("testing_crap.csv", index=None)
        print(raw_dataset)
        raw_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
        raw_dataset.save_to_disk(self.data_dir)


    #def setup(self, stage: str):
    def setup(self, stage=None):
        # https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html
        data = load_from_disk(self.data_dir)  # fs param will be removed by HF later on
        train_set_size = int(data.num_rows * 0.80)
        val_test_set_size = data.num_rows - train_set_size
        val_set_size = int(val_test_set_size * 0.65)
        test_set_size = val_test_set_size - val_set_size
        self.train_data, self.val_data, self.test_data = random_split(data, [train_set_size, val_set_size, test_set_size])
        print(f"total: {data.num_rows} | test {len(self.test_data)} | train {len(self.train_data)} | valid {len(self.val_data)}")
    
    #def train_dataloader(self) -> TRAIN_DATALOADERS:
    #    return super().train_dataloader()
    # https://www.geeksforgeeks.org/understanding-pytorch-lightning-datamodules/
    
    def train_dataloader(self) -> DataLoader:  # no num_workers. Before we parallelized?
        #return DataLoader(self.train_data)  #TODO: Pass workers as well and see if batches is best
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=6)  # TODO: try to set shuffle

    def val_dataloader(self) -> DataLoader:
        #return DataLoader(self.val_data)
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=6)

    def test_dataloader(self) -> DataLoader:
        #return DataLoader(self.test_data)
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=6)

    def preprocess_lists(self, df, headers):

        for h in headers:
            df[h] = df[h].apply(ast.literal_eval)
            df[h] = df[h].apply(self.sentinel_tkn.join)
        return df
    

    def preprocess_tokenize(self, dataset):

        #Dataset.from_pandas(df)  # no...
        # https://stackoverflow.com/questions/63017931/using-huggingface-trainer-with-distributed-data-parallel
        #tokenizer = T5TokenizerFast.from_pretrained("t5-small")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        
        ner = ["items: " + inp for inp in dataset["NER"]]

        ingredients = ["ingredients: " + inp for inp in dataset["ingredients"]] 
        #directions = [inp for inp in dataset["directions"]]  # LABELS
        directions = []
        for i, recipe in enumerate(dataset["title"]):
            lab = recipe + ': ' + dataset["directions"][i]
            directions.append(lab)

        model_inputs = tokenizer(ingredients, max_length=max_input_length,
                                 padding="max_length", truncation=True)
        labels = tokenizer(directions, max_length=max_input_length,
                           padding="max_length", truncation=True).input_ids
        

        labels_with_ignore_index = []
        for labels_example in labels:
            labels_example = [label if label != 0 else -100 for label in labels_example]
            labels_with_ignore_index.append(labels_example)
        
        model_inputs["labels"] = labels_with_ignore_index
        
        return model_inputs
        

        #print(model_inputs[0])  # ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
        #print(inputs[5])
        #print(model_inputs[5].ids)
        #test_str = tokenizer.decode(model_inputs[5].ids)
        #print('-'*8)
        #print(test_str)
        #print('-'*8)


        #test_str = tokenizer.convert_ids_to_tokens(model_inputs[0].ids)

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
