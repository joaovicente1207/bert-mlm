from transformers import RobertaTokenizer

vocab_file = './tokenizer/vocab.json'
merges_file = './tokenizer/merges.txt'
tokenizer = RobertaTokenizer(vocab_file=vocab_file, merges_file=merges_file, lowercase=True, add_prefix_space=True)

text = "mesa com cadeira 2m"

print(tokenizer.tokenize(text))
print(tokenizer.encode(text))
