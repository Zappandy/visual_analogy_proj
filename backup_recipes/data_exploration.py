#from transformers import FlaxAutoModelForSeq2SeqLM
#from transformers import AutoTokenizer
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from pytorch_lightning import LightningDataModule
#from datasets import Dataset  # install
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
        self.sentinel_tkn = "<eos>"
        self.sentinel_tkn = "<extra_id_99>"


    def prepare_data(self, test_size=0.2, dev_size=0.2):

        # https://github.com/huggingface/transformers/issues/16986
        df = pd.read_csv(self.data_dir)
        #df = pd.read_csv(self.data_dir, index_col=0)  # to clean weird idx
        #df.drop(["source", "link"], axis=1, inplace=True)  # only use with real_file
        #self.df.reset_index(drop=True, inplace=True)
        headers = ["ingredients", "NER", "directions"]  # title
        df = self.preprocess_lists(df, headers)

        # tokenize...
        self.preprocess_tokenize(df)
        raise SystemExit

        #df["tokenizer"].apply
        #TODO: After tokenization split with setup


    #TODO: to reload with GPU
    def setup(self):
        #train_df, test_df = train_test_split(df, test_size=test_size, random_state=SEED)
        #train_df, dev_df = train_test_split(train_df, test_size=dev_size, random_state=SEED)
        #return {"train": train_df, "test": test_df, "dev": dev_df}
        pass

    def preprocess_lists(self, df, headers):

        for h in headers:
            df[h] = df[h].apply(ast.literal_eval)
            df[h] = df[h].apply(self.sentinel_tkn.join)
        #TODO: JOIN EACH ELEMENT TO CREATE A STRING AND FACILITATE TOKENIZATION
        return df
    
    def vocab_augmentation(self, tokenizer):


        vocab = tokenizer.get_vocab()
        print(vocab["<extra_id_0>"])  # from 32099 to 32000 in descending order?
        tokenizer.vocab.pop("<extra_id_0>")

        vocab = tokenizer.get_vocab()
        print(vocab["<extra_id_0>"])  # from 32099 to 32000 in descending order?
        raise SystemExit
        inv_vocab = {v: k for k, v in vocab.items()}
        print(inv_vocab[32099])  # from 32099 to 32000 in descending order?
        tokenizer.add_tokens(self.sentinel_tkn)

        #tokenizer.add_tokens("tito", special_tokens=True)

        print(tokenizer.all_special_tokens)
        #print(vocab["<extra_id_0>"])  
        #for i in range(32099, 31999, -1):
        #    print(inv_vocab[i])
        raise SystemExit
        pass

    def preprocess_tokenize(self, df):

        #Dataset.from_pandas(df)  # no...
        tokenizer = T5TokenizerFast.from_pretrained("t5-small")  # size of vocab is the same for all
        
        self.vocab_augmentation(tokenizer)
        #df["ingredients"].apply(tokenizer.tokenize)
        #df["directions"].apply(tokenizer.tokenize)
        data = {header: df[header].values.tolist() for header in df.columns}
        prefix = "items: "
        # inputs are ingredients or NER?
        inputs = [prefix + inp for inp in data["NER"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length,
                                 padding="max_length", truncation=True)

        #print(model_inputs[0])  # ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
        print(inputs[5])
        print(model_inputs[5].ids)
        test_str = tokenizer.decode(model_inputs[5].ids)
        print('-'*8)
        print(test_str)
        print('-'*8)

        #for key, v in inv_vocab.items():
        #    print(key, v)
        #    break

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
