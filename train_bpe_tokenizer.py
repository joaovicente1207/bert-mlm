from tokenizers import ByteLevelBPETokenizer

paths = [f"data/corpus_{split}.txt" for split in ["train", "dev"]]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=1000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
#Save the Tokenizer to disk
tokenizer.save_model('./tokenizer')

#provavelmente vocab size tem que ser maior que o max len