from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader, random_split

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
    
        self.sos_token = self.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = self.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = self.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(f"Sequence length {self.seq_len} is too small for the given input.")
        
        encoder_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(enc_input_tokens, dtype=torch.int64), 
                self.eos_token, 
                torch.tensor([self.pad_token.item()] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        decoder_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(dec_input_tokens, dtype=torch.int64), 
                torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64), 
                self.eos_token, 
                torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        assert encoder_input.shape[0] == self.seq_len, f"Encoder input length {encoder_input.shape[0]} does not match seq_len {self.seq_len}"
        assert decoder_input.shape[0] == self.seq_len, f"Decoder input length {decoder_input.shape[0]} does not match seq_len {self.seq_len}"
        assert label.shape[0] == self.seq_len, f"Label length {label.shape[0]} does not match seq_len {self.seq_len}"
        
        return {
            "encoder_input": encoder_input, # shape: (seq_len,)
            "decoder_input": decoder_input, # shape: (seq_len,)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # shape: (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)) , # shape: (1, seq_len) & (1, seq_len, seq_len)
            "label": label, # shape: (seq_len,)
            "src_text": src_text,
            "tgt_text": tgt_text
        }
        
def casual_mask(size):
    mask = torch.triu(torch.ones(1,size, size), diagonal=1).type(torch.bool) # shape: (1, 1, size, size)
    return mask == 0
        