from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['painel acab linear h25 680mm - ouro bco',
             'painel editavel de 25 950 x 530 x 25']

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("./tokenizer", max_len=512)

model = AutoModel.from_pretrained('./bert_prod')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print("Sentence embeddings:")
print(sentence_embeddings)

from torch.nn import CosineSimilarity

cos = CosineSimilarity(dim=0, eps=1e-6)
output = cos(sentence_embeddings[0], sentence_embeddings[1])

print(output)
