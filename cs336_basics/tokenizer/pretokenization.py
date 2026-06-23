import os
from typing import BinaryIO
import regex as re


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    # chunk_boundaries(guessly) → [0, chunk_size, 2*chunk_size, 3*chunk_size, file_size]
    
    mini_chunk_size = 4096  # 以 4KB 为步长预读

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be <= desired_num_chunks
    return sorted(set(chunk_boundaries))
    # chunk_boundaries(properly) → [0, chunk_size+found_at1, 2*chunk_size+found_at2, ..., file_size]，共 <=desired_num_chunks 个区间
    # → [0, "<|endoftext|>"_begin_for_chunk1, "<|endoftext|>"_begin_for_chunk2, ..., file_size]
    # 除了第一块,每一块都是以 <|endoftext|> 开头的。也就是说 special token 归属于"它后面那一块",而不是前面那块。

## Usage
def build_freq_table(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    freq_table: dict[tuple[bytes, ...], int] = {}
    
    with open(input_path, "rb") as f:
        num_processes = 4
        # 分块只需挑一个 token 当"够用的切刀" → "<|endoftext|>"
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        # TODO: 并行化
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""        # GPT-2 like
            
            # Removing special tokens before pre-tokenization
            # 块内部要尊重全部 special token,因为它们都是不可跨越的语义边界、要以它们为界来分别对每个段文本统计。
            escaped = [re.escape(st) for st in special_tokens]
            chunk_texts = re.split(pattern='|'.join(escaped), string=chunk)
            # 统计
            for text in chunk_texts:
                for pre_token in re.finditer(PAT, text):
                    single_bytes = pre_token.group().encode('utf-8', errors="ignore")
                    bytes_tuple = tuple(bytes([x]) for x in list(single_bytes))    # b(ytes style)token
                    freq_table[bytes_tuple] = freq_table.get(bytes_tuple, 0) + 1
    
    return freq_table