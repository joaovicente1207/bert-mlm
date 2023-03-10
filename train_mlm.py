from transformers import RobertaConfig
from transformers import RobertaForMaskedLM

config = RobertaConfig(
    vocab_size=100,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
model = RobertaForMaskedLM(config=config)
print(model.num_parameters())


from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("./tokenizer", max_len=512)

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./data/corpus_train.txt",
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./model",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=100,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("./bert_prod")