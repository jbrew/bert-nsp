import runway
import torch
from transformers import BertTokenizer
from transformers import BertForNextSentencePrediction
from runway.data_types import array, text, number, boolean


# Setup block copy-pasted from Cris's tutorial
@runway.setup(options={"checkpoint": runway.category(description="Pretrained checkpoints to use.",
                                      choices=['celebAHQ-512', 'celebAHQ-256', 'celeba'],
                                      default='celebAHQ-512')})
def setup(opts):
    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer


@runway.command(name='sequence_score',
               inputs={ 'sentence1': text(), 'sentence2': text()},
               outputs={ 'score': number()})
def sequence_score(setup_tuple, inputs):
    model, tokenizer = setup_tuple
    combined_sentences = sentence1 + ' ' + sentence2      # may be better to concatenate *after* tokenization using special [SEP] token
    input_tokens = tokenizer.encode(combined, add_special_tokens=True)
    input_ids = torch.tensor(input_tokens).unsqueeze(0)
    outputs = model(input_ids)
    seq_relationship_scores = outputs[0]     # outputs is an array with losses as the first value and logits as the second
    return seq_relationship_scores

if __name__ == '__main__':
    runway.run()