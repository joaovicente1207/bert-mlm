from tokenizers import CharBPETokenizer

paths = [f"data/corpus_{split}.txt" for split in ["train", "dev"]]

# Initialize a tokenizer
tokenizer = CharBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=100, min_frequency=1, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
#Save the Tokenizer to disk
tokenizer.save_model('./tokenizer')

#provavelmente vocab size tem que ser maior que o max len