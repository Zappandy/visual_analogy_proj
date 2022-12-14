from transformers import FlaxAutoModelForSeq2SeqLM
from transformers import AutoTokenizer

MODEL_NAME_OR_PATH = "flax-community/t5-recipe-generation"

prefix = "items: "
# generation_kwargs = {
#     "max_length": 512,
#     "min_length": 64,
#     "no_repeat_ngram_size": 3,
#     "early_stopping": True,
#     "num_beams": 5,
#     "length_penalty": 1.5,
# }
generation_kwargs = {
    "max_length": 512,
    "min_length": 64,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
    "top_k": 60,
    "top_p": 0.95
}


tokens_map = {
    "<sep>": "--",
    "<section>": "\n"
}

class RecipeGenerator:

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
        self.special_tokens = self.tokenizer.all_special_tokens

    def generation_function(self, texts):

        model = FlaxAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)

        _inputs = texts if isinstance(texts, list) else [texts]
        inputs = [prefix + inp for inp in _inputs]
        inputs = self.tokenizer(
            inputs, 
            max_length=256, 
            padding="max_length", 
            truncation=True, 
            return_tensors="jax"
        )
    
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
    
        output_ids = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            **generation_kwargs
        )
        generated = output_ids.sequences
        generated_recipe = self.target_postprocessing(
            self.tokenizer.batch_decode(generated, skip_special_tokens=False))

        return generated_recipe

    def skip_special_tokens(self, text):
        for token in self.special_tokens:
            text = text.replace(token, "")
    
        return text
    
    def target_postprocessing(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
    
        new_texts = []
        for text in texts:
            text = self.skip_special_tokens(text)
    
            for k, v in tokens_map.items():
                text = text.replace(k, v)
    
            new_texts.append(text)
    
        return new_texts


def display_recipes(items):
    recipe_generator = RecipeGenerator()
    generated = recipe_generator.generation_function(items)
    recipes = []
    for text in generated:
        message = ''
        sections = text.split("\n")
        for section in sections:
            section = section.strip()
            if section.startswith("title:"):
                section = section.replace("title:", "")
                headline = "TITLE"
            elif section.startswith("ingredients:"):
                section = section.replace("ingredients:", "")
                headline = "INGREDIENTS"
            elif section.startswith("directions:"):
                section = section.replace("directions:", "")
                headline = "DIRECTIONS"
    
            if headline == "TITLE":
                head = f"[{headline}]: {section.strip().capitalize()}"
                #print(f"[{headline}]: {section.strip().capitalize()}")
                print(head)
                message += head + '\n'
            else:
                section_info = [f"  - {i+1}: {info.strip().capitalize()}" for i, info in enumerate(section.split("--"))]
                headline = f"[{headline}]"
                section = "\n".join(section_info)
                message += headline + '\n' + section + '\n'
                print(f"[{headline}]:")
                print("\n".join(section_info))
    
        print("-" * 130)
        message += ('\n' + '-' * 130)
        recipes.append(message)
    return recipes    
