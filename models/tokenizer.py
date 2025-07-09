import torch
import logging

# Set up logging
logger = logging.getLogger("models/tokenizer.py")

class BytePairEncoder:
    """
    임시 구현: 간단한 문자 기반 토크나이저
    실제 BPE는 나중에 구현할 예정
    """
    def __init__(self, device=None):
        self.device = device if device else torch.device("cpu")
        
        # 특수 토큰들
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.sep_token = "<SEP>"
        
        # 특수 토큰 ID들
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.sep_token_id = 4
        
        # 기본 어휘 구축 (ASCII 문자들 + 특수 토큰들)
        self.char_to_id = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id,
            self.sep_token: self.sep_token_id,
        }
        
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        
        # ASCII 문자들 추가 (공백, 영문자, 숫자, 기본 특수문자)
        current_id = 5
        for i in range(32, 127):  # 인쇄 가능한 ASCII 문자들
            char = chr(i)
            if char not in self.char_to_id:
                self.char_to_id[char] = current_id
                self.id_to_char[current_id] = char
                current_id += 1
        
        self.vocab_size = len(self.char_to_id)  # 어휘 크기

        logger.info(f"Simple tokenizer initialized with {len(self.char_to_id)} characters")

    def encode(self, text):
        """텍스트를 토큰 ID 텐서로 변환"""
        if isinstance(text, str):
            # 문자열을 문자 단위로 토큰화
            token_ids = []
            for char in text:
                token_id = self.char_to_id.get(char, self.unk_token_id)
                token_ids.append(token_id)
            
            return torch.tensor(token_ids, device=self.device, dtype=torch.long)
        else:
            raise ValueError("Input must be a string")

    def decode(self, tokens):
        """토큰 ID 텐서를 텍스트로 변환"""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy().tolist()
        
        if isinstance(tokens[0], list):  # 배치 처리
            return [self._decode_single(token_list) for token_list in tokens]
        else:
            return self._decode_single(tokens)
    
    def _decode_single(self, token_ids):
        """단일 토큰 시퀀스 디코딩"""
        chars = []
        for token_id in token_ids:
            if token_id in self.id_to_char:
                char = self.id_to_char[token_id]
                # 특수 토큰들은 건너뛰기 (실제 텍스트에서 제외)
                if char not in [self.pad_token, self.bos_token, self.eos_token, self.sep_token]:
                    chars.append(char)
            else:
                chars.append(self.unk_token)
        
        return ''.join(chars)
    
    def get_vocab_size(self):
        """어휘 크기 반환"""
        return len(self.char_to_id)