from transformers import pipeline

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("./tokenizer", max_len=512)


fill_mask = pipeline(
    "fill-mask",
    model="./bert_prod",
    tokenizer = tokenizer
)

print(fill_mask("cadeira 2 <mask>"))

