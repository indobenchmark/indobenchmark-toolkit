# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
""" Tokenization classes for IndoNLG model."""

import os
from shutil import copyfile
from typing import List, Optional, Tuple
from transformers import PreTrainedTokenizer

import sentencepiece as spm

from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "indobenchmark/indobart": "https://huggingface.co/indobenchmark/indobart/resolve/main/sentencepiece.bpe.model",
        "indobenchmark/indogpt": "https://huggingface.co/indobenchmark/indogpt/resolve/main/sentencepiece.bpe.model",
        "indobenchmark/indobart-v2": "https://huggingface.co/indobenchmark/indobart-v2/resolve/main/sentencepiece.bpe.model"
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "indobenchmark/indobart": 768,
    "indobenchmark/indogpt": 768,
    "indobenchmark/indobart-v2": 768
}

SHARED_MODEL_IDENTIFIERS = [
    # Load with
    "indobenchmark/indobart",
    "indobenchmark/indogpt",
    "indobenchmark/indobart-v2"
]

SPIECE_UNDERLINE = "▁"

class IndoNLGTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids","attention_mask"]
    input_error_message = "text input must of type `str` (single example), `List[str]` (batch of examples)."

    def __init__(
        self,
        vocab_file,
        decode_special_token=True,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        additional_special_tokens=["[java]","[sunda]","[indonesia]","<mask>"],
        **kwargs
    ):
        super().__init__(
            vocab_file=vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file
        self.decode_special_token = decode_special_token
        self.model_max_length = 1024
        
        # HACK: These tokens were added by fairseq but don't seem to be actually used when duplicated in the actual
        # sentencepiece vocabulary (this is the case for <s> and </s>
        self.special_tokens_to_ids = {
            "[javanese]": 40000, 
            "[sundanese]": 40001, 
            "[indonesian]": 40002,
            "<mask>": 40003
        }
        self.special_ids_to_tokens = {v: k for k, v in self.special_tokens_to_ids.items()}
        
        # Store Language token ID
        self.javanese_token = '[javanese]'
        self.javanese_token_id = 40000
        self.sundanese_token = '[sundanese]'
        self.sundanese_token_id = 40001
        self.indonesian_token = '[indonesian]'
        self.indonesian_token_id = 40002
        
        self.special_token_ids = [
            self.bos_token_id, self.eos_token_id, self.sep_token_id, self.cls_token_id, 
            self.unk_token_id, self.pad_token_id, self.mask_token_id,
            self.javanese_token_id, self.sundanese_token_id, self.indonesian_token_id
        ]
    
    def prepare_input_for_generation(self, inputs, model_type='indobart', lang_token='[indonesian]', decoder_inputs=None,
                                             decoder_lang_token='[indonesian]', padding='longest', return_tensors=None):
        """
        Build model inputs for a specified `model_type`. There are two possible `model_type`, i.e., indobart and indogpt.
        
        When `model_type` is indogpt, `lang_token`, `decoder_inputs`, and `decoder_lang_token` parameters will be ignored 
        and the input will be encoded in the gpt2 sequence format as follow: 
        
        - indogpt sequence: ``<s> X``
        
        When `model_type` is indobart, `inputs` and `lang_token` are used as the sequence and language identifier for the indobart encoder, 
        while `decoder_inputs` and `decoder_lang_token` are used as the sequence and language identifier of the decoder
        
        - indobart encoder sequence: ``X </s> <lang_token_id>``
        - indobart decoder sequences: ``<decoder_lang_token_id> X </s>``

        Args:
            inputs (:obj:`str` or `List[str]`):
                text sequence or list of text sequences to be tokenized.
            model_type (:obj:`str`, defaults to :obj:`indobart`):
                model type to determine the format of the tokenized sequence. Valid values are `indobart` and `indogpt`.
            lang_token (:obj:`str`, defaults to :obj:`[indonesian]`):
                language token to determine the format of the tokenized sequence. Valid values are `[indonesian]`, `[sundanese], and [javanese]`.
            decoder_inputs (:obj:`str` or `List[str]`, `optional`):
                decoder text sequence or list of text sequences to be tokenized.
            decoder_lang_token (:obj:`str`, defaults to :obj:`[indonesian]`):
                decoder language token to determine the format of the tokenized sequence. Valid values are `[indonesian]`, `[sundanese], and [javanese]`.
            padding (:obj:`str`, defaults to :obj:`longest`):
                padding strategy to pad the tokenized sequences. Valid values are `longest`, `max_length`, and `do_not_pad`.
            return_tensors (:obj:`str`, defaults to :obj:`None`):
                Returned tensor type of the tokenized sequence. When set to `None`, the return type will be List[int]. Valid values are `None`, `pt`, and `tf`

        Returns:
            :obj:`Dict`: Dictionary with `input_ids`, `attention_mask`, `decoder_input_ids` (optional), and `decoder_attention_mask` (optional)
        """        
        if model_type == 'indogpt':
            # Process indogpt input
            if type(inputs) == str:
                 return self(f'<s> {inputs}', padding=padding, return_tensors=return_tensors)
            elif type(inputs) == list:
                if len(inputs) == 0 or type(inputs[0]) != str:
                    raise ValueError(IndoNLGTokenizer.input_error_message)
                else:
                    return self([f'<s> {input_data}' for input_data in inputs], padding=padding, return_tensors=return_tensors)
            else:
                raise ValueError(IndoNLGTokenizer.input_error_message)
        elif model_type == 'indobart':
                                     
            # Process encoder input
            if lang_token not in self.special_tokens_to_ids:
                raise ValueError(f"Unknown lang_token `{lang_token}`, lang_token must be either `[javanese]`, `[sundanese]`, or `[indonesian]`")  
            elif type(inputs) == list:
                if len(inputs) == 0 or type(inputs[0]) != str:
                    raise ValueError(IndoNLGTokenizer.input_error_message)
            elif type(inputs) != str:
                raise ValueError(IndoNLGTokenizer.input_error_message)
                
            lang_id = self.special_tokens_to_ids[lang_token]
            input_batch = self(inputs, return_attention_mask=False)
            if type(inputs) == str:
                input_batch['input_ids'] = [self.bos_token_id] + input_batch['input_ids'] + [self.eos_token_id, lang_id]
            else:
                input_batch['input_ids'] = list(map(lambda input_ids: [self.bos_token_id] + input_ids + [self.eos_token_id, lang_id], input_batch['input_ids']))
            
            if decoder_inputs is None:
                # Return encoder input
                return self.pad(input_batch, return_tensors=return_tensors)
            else:
                # Process decoder input
                if decoder_lang_token not in self.special_tokens_to_ids:
                    raise ValueError(f"Unknown decoder_lang_token `{decoder_lang_token}`, decoder_lang_token must be either `[javanese]`, `[sundanese]`, or `[indonesian]`")  
                elif type(decoder_inputs) == list:
                    if len(decoder_inputs) == 0:
                        raise ValueError(IndoNLGTokenizer.input_error_message)
                    elif type(decoder_inputs[0]) != str:
                        raise ValueError(IndoNLGTokenizer.input_error_message)
                elif type(decoder_inputs) != str:
                    raise ValueError(IndoNLGTokenizer.input_error_message)

                decoder_lang_id = self.special_tokens_to_ids[decoder_lang_token]
                decoder_input_batch = self(decoder_inputs, return_attention_mask=False)
                
                if type(decoder_inputs) == str:
                    decoder_input_batch['input_ids'] = [lang_id, self.bos_token_id] + decoder_input_batch['input_ids'] + [self.eos_token_id]
                else:
                    decoder_input_batch['input_ids'] = list(map(lambda input_ids: [lang_id, self.bos_token_id] + input_ids + [self.eos_token_id], decoder_input_batch['input_ids']))
                # Padding
                input_batch = self.pad(input_batch, return_tensors=return_tensors)
                decoder_input_batch = self.pad(decoder_input_batch, return_tensors=return_tensors)
                
                # Store into a single dict
                input_batch['decoder_input_ids'] = decoder_input_batch['input_ids']
                input_batch['decoder_attention_mask'] = decoder_input_batch['attention_mask']
                
                return input_batch

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    @property
    def vocab_size(self):
        return 4 + len(self.sp_model)

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(text.lower(), out_type=str)
    
    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.special_tokens_to_ids:
            return self.special_tokens_to_ids[token]
        return self.sp_model.PieceToId(token)
    
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if not self.decode_special_token and index in self.special_token_ids:
            return ''
            
        if index in self.special_ids_to_tokens:
            return self.special_ids_to_tokens[index]
        
        token = self.sp_model.IdToPiece(index)
        if '<0x' in token:
            char_rep = chr(int(token[1:-1], 0))
            if char_rep.isprintable():
                return char_rep
        return token
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def decode(self, inputs, skip_special_tokens=False):
        prev_val = self.decode_special_token
        self.decode_special_token = not skip_special_tokens
        
        outputs = super().decode(inputs, skip_special_tokens=False)
        self.decode_special_token = prev_val
        
        return outputs.replace('▁','').strip()
        