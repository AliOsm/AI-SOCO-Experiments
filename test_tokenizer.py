from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import RobertaProcessing


tokenizer = ByteLevelBPETokenizer(
    'tokenizer/vocab.json',
    'tokenizer/merges.txt',
)

tokenizer._tokenizer.post_processor = RobertaProcessing(
    ('</s>', tokenizer.token_to_id('</s>')),
    ('<s>', tokenizer.token_to_id('<s>')),
)

tokenizer.enable_truncation(max_length=512)
tokenizer.enable_padding(pad_id=tokenizer.token_to_id('<pad>'), pad_token='<pad>', length=512)

code = '#include<iostream>'

print(code)
print(tokenizer.encode(code).tokens)
