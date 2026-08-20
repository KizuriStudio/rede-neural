import numpy as np


class TextDataset:
    """Dataset de texto otimizado pra pouca memória."""

    def __init__(self, text, seq_length=64, max_text_size=500000):
        if len(text) > max_text_size:
            text = text[:max_text_size]

        self.text = text
        self.seq_length = seq_length
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

        if self.vocab_size <= 256:
            self.encoded = np.array([self.char_to_idx[c] for c in text], dtype=np.uint8)
        else:
            self.encoded = np.array([self.char_to_idx[c] for c in text], dtype=np.int32)

    def __len__(self):
        return max(0, len(self.encoded) - self.seq_length - 1)

    def __getitem__(self, idx):
        x = self.encoded[idx:idx + self.seq_length].astype(np.int32)
        y = self.encoded[idx + 1:idx + self.seq_length + 1].astype(np.int32)
        return x, y

    def encode(self, text):
        return np.array([self.char_to_idx.get(c, 0) for c in text], dtype=np.int32)

    def decode(self, indices):
        return "".join([self.idx_to_char.get(int(i), "?") for i in indices])


class DataLoader:
    """DataLoader com sampling aleatório."""

    def __init__(self, dataset, batch_size=32, shuffle=True, samples_per_epoch=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n = len(dataset)
        self.samples_per_epoch = samples_per_epoch or min(self.n, 5000)

    def __iter__(self):
        indices = np.random.randint(0, self.n, size=self.samples_per_epoch)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            if len(batch_idx) < self.batch_size:
                continue

            xs = np.zeros((len(batch_idx), self.dataset.seq_length), dtype=np.int32)
            ys = np.zeros((len(batch_idx), self.dataset.seq_length), dtype=np.int32)

            for i, idx in enumerate(batch_idx):
                x, y = self.dataset[int(idx)]
                xs[i] = x
                ys[i] = y

            yield xs, ys

    def __len__(self):
        return self.samples_per_epoch // self.batch_size
