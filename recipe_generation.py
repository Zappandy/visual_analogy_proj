from transformers import T5ForConditionalGeneration, T5Tokenizer
from demo.utils import generation_kwargs
import torch

def generate_recipes(test_dataset):
    model = T5ForConditionalGeneration.from_pretrained('./stored_models/ruby_code_model')
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    #test_example = next(iter(data_reader.test_data))
    test_example = next(iter(test_dataset))
    
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
