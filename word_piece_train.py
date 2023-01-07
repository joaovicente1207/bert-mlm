from tokenizers import trainers
from tokenizers import Tokenizer
from tokenizers.models import WordPiece

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]",))

trainer = trainers.WordPieceTrainer(vocab_size=100,min_frequency=1, 
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()

files = [f"data/corpus_{split}.txt" for split in ["train", "dev"]]
tokenizer.train(files, trainer)

tokenizer.save("wordpiece.json")