---
date: '2026-01-04'
draft: false
title: 'Building an LLM from Scratch: CS336 Assignment 1'
description: 'My implementation journey through Stanford CS336 Assignment 1 - building language model components from the ground up.'
tags: ['CS336', 'LLM', 'Deep Learning', 'Transformers', 'NLP']
math: true
comments: true
---

This post documents my journey through **Stanford CS336: Language Models from Scratch** — specifically Assignment 1, where we implement core components of a language model.

## Overview

CS336 is Stanford's deep dive into building large language models from first principles. Assignment 1 focuses on:

- Tokenization (BPE implementation)
- Transformer architecture components
- Training loop fundamentals

## Key Takeaways

### 1. Byte-Pair Encoding (BPE)

*Coming soon...*

### 2. Attention Mechanism

*Coming soon...*

### 3. Training Considerations

*Coming soon...*

## Implementation Highlights

```python
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

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

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

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
```

## Challenges & Lessons Learned

*Coming soon...*

## References

1. [Stanford CS336 Course Page](https://stanford-cs336.github.io/spring2025/)
2. Vaswani et al., "Attention Is All You Need" (2017)
3. Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (2016)

---

*This post is a work in progress. Check back for updates!*
