# ==============================================================================
# 文件: bpe.py (你需要自己创建这个文件)
# ==============================================================================
import collections
import re

class BpeTokenizer:
    """
    一个从零开始实现的 BPE 分词器。
    内部处理全部基于 bytes，以保证通用性。
    """
    def __init__(self):
        # 核心数据结构
        self.vocab: dict[bytes, int] = {}  # token (bytes) -> id (int)
        self.merges: list[tuple[bytes, bytes]] = []  # 有序的合并规则
        
        # 用于解码的辅助数据结构
        self.inverse_vocab: dict[int, bytes] = {}
        
        # 特殊 token 的处理
        self.special_tokens: set[bytes] = set()
        self.special_token_pattern: re.Pattern | None = None

    def _build_special_token_pattern(self):
        """根据特殊 token 构建用于切分文本的正则表达式。"""
        # 对特殊 token 进行转义，以防它们包含 regex 特殊字符
        escaped_tokens = [re.escape(token.decode('utf-8')) for token in self.special_tokens]
        if escaped_tokens:
            # 按长度降序排序，以优先匹配最长的 token (例如 <|endoftext|> vs <|end|>)
            escaped_tokens.sort(key=len, reverse=True)
            pattern_str = "|".join(escaped_tokens)
            self.special_token_pattern = re.compile(pattern_str)

    def get_pair_freqs(self, splits: list[list[bytes]]) -> collections.defaultdict[tuple[bytes, bytes], int]:
        """统计所有相邻符号对的频率。"""
        pair_freqs = collections.defaultdict(int)
        for split in splits:
            for i in range(len(split) - 1):
                pair = (split[i], split[i+1])
                pair_freqs[pair] += 1
        return pair_freqs

    def merge_pair(self, split: list[bytes], pair_to_merge: tuple[bytes, bytes]) -> list[bytes]:
        """在单个单词的符号列表中执行一次合并操作。"""
        new_split = []
        i = 0
        while i < len(split):
            if i < len(split) - 1 and (split[i], split[i+1]) == pair_to_merge:
                new_split.append(split[i] + split[i+1])
                i += 2
            else:
                new_split.append(split[i])
                i += 1
        return new_split

    def train(self, corpus: str, vocab_size: int):
        """
        BPE 训练算法。
        """
        if vocab_size < 256:
            raise ValueError("词汇表大小必须至少为 256 以覆盖所有字节。")

        # 1. 初始化
        # 初始词汇表是所有单个字节
        initial_vocab = {bytes([i]): i for i in range(256)}
        self.vocab = initial_vocab
        
        # 将语料库预切分为单词，并将每个单词转换为字节列表的列表
        encoded_corpus = corpus.encode('utf-8')
        # 这里使用一个简单的 regex 来切分单词和空格/标点
        # GPT-2 使用了更复杂的 regex
        words = re.findall(rb"\w+|[^\w\s]", encoded_corpus)
        word_splits = [[byte.to_bytes(1, 'big') for byte in word] for word in words]

        # 2. 迭代合并
        num_merges = vocab_size - 256
        for i in range(num_merges):
            pair_freqs = self.get_pair_freqs(word_splits)
            if not pair_freqs:
                break
            
            best_pair = max(pair_freqs, key=pair_freqs.get)
            new_token = best_pair[0] + best_pair[1]
            
            # 更新词汇表和合并规则
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
                self.merges.append(best_pair)

            # 在语料库中执行合并
            word_splits = [self.merge_pair(split, best_pair) for split in word_splits]
        
        # 训练完成后，构建解码用的反向词汇表
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        """
        将文本字符串编码为 token ID 列表。
        """
        if not self.special_token_pattern:
            self._build_special_token_pattern()

        token_ids = []
        
        # 首先，使用特殊 token 的 regex 来切分文本
        # 这会把文本切分成 "普通文本" 和 "特殊 token" 的交替序列
        if self.special_token_pattern:
            chunks = self.special_token_pattern.split(text)
            special_tokens_found = self.special_token_pattern.findall(text)
        else:
            chunks = [text]
            special_tokens_found = []

        # 交错处理普通文本和特殊 token
        for i, chunk in enumerate(chunks):
            # 处理普通文本块
            if chunk:
                encoded_chunk = chunk.encode('utf-8')
                # 简单地按单词切分
                words = re.findall(rb"\w+|[^\w\s]", encoded_chunk)
                for word in words:
                    split = [byte.to_bytes(1, 'big') for byte in word]
                    
                    # 贪婪地、按顺序应用所有已学习的合并规则
                    for pair in self.merges:
                        split = self.merge_pair(split, pair)
                    
                    # 将最终的符号序列映射到 ID
                    for token_bytes in split:
                        token_ids.append(self.vocab[token_bytes])

            # 处理特殊 token
            if i < len(special_tokens_found):
                special_token_bytes = special_tokens_found[i].encode('utf-8')
                token_ids.append(self.vocab[special_token_bytes])
        
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """
        将 token ID 列表解码回文本字符串。
        """
        tokens_bytes = [self.inverse_vocab[token_id] for token_id in token_ids]
        full_bytes = b"".join(tokens_bytes)
        return full_bytes.decode('utf-8', errors='replace')
    

    