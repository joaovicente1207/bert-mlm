from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE())

from tokenizers.pre_tokenizers import Whitespace

tokenizer.pre_tokenizer = Whitespace()

from tokenizers.trainers import BpeTrainer
trainer = BpeTrainer(vocab_size=100, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])



files = [f"data/corpus_{split}.txt" for split in ["train", "dev"]]
tokenizer.train(files=files, trainer=trainer)

tokenizer.save("tokenizer/tokenizer-prod.json")


output = tokenizer.encode("cadeira de praia amarela")
print(output.tokens)