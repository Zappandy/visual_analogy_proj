#from transformers import FlaxAutoModelForSeq2SeqLM
#from transformers import AutoTokenizer
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from pytorch_lightning import LightningDataModule
from datasets import Dataset
from transformers import T5TokenizerFast  # no Fast method?


SEED = 42
path = "data/dataset/full_dataset.csv"
test_path = "./example_recipes.csv"
max_input_length = 128

class RecipeTXTData(LightningDataModule):

    #def __init__(self, data_dir: str=path):
    def __init__(self, data_dir: str=test_path):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = None
        self.sentinel_tkn = "<extra_id_99>"


    def prepare_data(self):

        # https://github.com/huggingface/transformers/issues/16986
        df = pd.read_csv(self.data_dir)
        # DO NOT DELETE THESE 3 LINES ARE FOR FULL DATASET
        #df = pd.read_csv(self.data_dir, index_col=0)  # to clean weird idx
        #df.drop(["source", "link"], axis=1, inplace=True)  # only use with real_file
        #self.df.reset_index(drop=True, inplace=True)
        headers = ["ingredients", "NER", "directions"]  # title
        df = self.preprocess_lists(df, headers)

        # tokenize...
        raw_dataset = Dataset.from_pandas(df)
        self.dataset = raw_dataset.map(self.preprocess_tokenize, batched=True)



    #TODO: to reload with GPU
    def setup(self, stage: str):
        #train_df, test_df = train_test_split(df, test_size=test_size, random_state=SEED)
        #train_df, dev_df = train_test_split(train_df, test_size=dev_size, random_state=SEED)
        #return {"train": train_df, "test": test_df, "dev": dev_df}
        #TODO: STORE TOK OR DATASET IN DISK THEN RELOAD?
        # https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html
        if stage == "fit":
            self.train_data
            self.val_data
        pass

    def preprocess_lists(self, df, headers):

        for h in headers:
            df[h] = df[h].apply(ast.literal_eval)
            df[h] = df[h].apply(self.sentinel_tkn.join)
        #TODO: JOIN EACH ELEMENT TO CREATE A STRING AND FACILITATE TOKENIZATION
        return df
    

    def preprocess_tokenize(self, dataset):

        #Dataset.from_pandas(df)  # no...
        tokenizer = T5TokenizerFast.from_pretrained("t5-small")
        
        ner = ["items: " + inp for inp in dataset["NER"]]

        ingredients = ["ingredients: " + inp for inp in dataset["ingredients"]] 
        directions = [inp for inp in dataset["directions"]]  # LABELS
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
