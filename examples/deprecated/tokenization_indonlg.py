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
from transformers import CamembertTokenizer

import sentencepiece as spm

from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "indobart": "https://huggingface.co/indobart/resolve/main/sentencepiece.bpe.model",
        "indogpt": "https://huggingface.co/indogptresolve/main/sentencepiece.bpe.model",
        "indobart-v2": "https://huggingface.co/indobart-v2/resolve/main/sentencepiece.bpe.model"
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "indobenchmark/indobart": 768,
    "ndobenchmark/indogpt": 768,
    "indobenchmark/indobart-v2": 768
}

SHARED_MODEL_IDENTIFIERS = [
    # Load with
    "indobenchmark/indobart",
    "indobenchmark/indogpt",
    "indobenchmark/indobart-v2"
]

SPIECE_UNDERLINE = "▁"

class IndoNLGTokenizer(CamembertTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids","attention_mask"]

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
        additional_special_tokens=["[java]","[sunda]","[indo]","<mask>"],
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
        
        return self.sp_model.IdToPiece(index)

    def decode(self, inputs, skip_special_tokens=False):
        prev_val = self.decode_special_token
        self.decode_special_token = not skip_special_tokens
        
        outputs = super().decode(inputs, skip_special_tokens=skip_special_tokens)
        self.decode_special_token = prev_val
        
        return outputs
        