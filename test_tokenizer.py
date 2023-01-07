from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("./tokenizer", max_len=512)

text = "lm led tl slim 10 sobrepor autovolt 6500k branca-der/01 branca-7897079061215- cod.fci 4a8bfb1e-b524-4ff0-bf44-366b054ab8"

print(tokenizer.tokenize(text))
print('lenght: ', len(tokenizer.tokenize(text)))
print(tokenizer.encode(text))
