from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("./tokenizer", max_len=512)

text = "lm led tl slim 10 <mask> autovolt 6500k branca"

print(tokenizer.tokenize(text))
print('lenght: ', len(tokenizer.tokenize(text)))
print(tokenizer.encode(text))
