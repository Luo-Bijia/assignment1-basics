import os
from .pretokenization import build_freq_table

def convert_to_pairs(freq_table: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, ...], int]:
    # 将 freq_table 转换成连续字节对的频率表
    pairs_table: dict[tuple[bytes, ...], int] = {}
    for btoken, count in freq_table.items():
        for b1, b2 in zip(btoken, btoken[1:]):
            pairs_table[(b1, b2)] = pairs_table.get((b1, b2), 0) + count
    return pairs_table

def update_freq_table(freq_table: dict[tuple[bytes, ...], int], pair: tuple[bytes, bytes]) -> dict[tuple[bytes, ...], int]:
    # 将每个 token 下符合 pair 的两个 bytes 拼接成单个 bytes 对象
    new_freq_table = {}
    merged = pair[0] + pair[1]

    for btoken, count in freq_table.items():
        new_btoken = []
        i = 0
        while i < len(btoken):
            if i + 1 < len(btoken) and btoken[i] == pair[0] and btoken[i + 1] == pair[1]:
                new_btoken.append(merged)
                i += 2
            else:
                new_btoken.append(btoken[i])
                i += 1
        new_btoken = tuple(new_btoken)
        new_freq_table[new_btoken] = new_freq_table.get(new_btoken, count)
    return new_freq_table

def find_pair(pairs_table: dict[tuple[bytes, ...], int]) -> tuple[bytes, bytes]:
    # 返回 pairs_table 中最高频且符合字典序更大的字节对
    # TODO: 目前是O(n)复杂度  → 预期降到O(logn) → heap？
    pair = max(pairs_table.items(), key=lambda item: (item[1], item[0]))[0]
    return pair

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    freq_table = build_freq_table(input_path, special_tokens)   # pre-tokenization
    
    # Initalize vocab
    for id in range(256):   
        vocab[id] = vocab.get(id, bytes([id]))
        
    # merge
    num_merges = vocab_size - 256 - len(special_tokens)
    for i in range(num_merges):
        pairs_table = convert_to_pairs(freq_table)
        pair = find_pair(pairs_table)
        idx = 256 + i
        vocab[idx] = vocab.get(idx, pair[0] + pair[1])
        
        freq_table = update_freq_table(freq_table, pair)    # 旧 table 会被自动释放，新 table 存活
        merges.append(pair)

    # add special tokens to vocab's tail
    sp_off = len(vocab)
    for id in range(len(special_tokens)):
        vocab[sp_off + id] = vocab.get(sp_off + id, special_tokens[id].encode('utf-8', errors="ignore"))

    return (vocab, merges)