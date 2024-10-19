from torch.utils.data import Sampler
import random


class ConsecutiveBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, seq_len, drop_last=False, shuffle=True):
        super(ConsecutiveBatchSampler, self).__init__(data_source)
        self.data_source = data_source
        self.seq_len = seq_len
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        data_size = len(self.data_source)
        start_indices = [i for i in range(data_size - self.seq_len + 1)]

        if self.shuffle:
            random.shuffle(start_indices)

        batches = []
        for start_idx in start_indices:
            if len(batches) == self.batch_size:
                yield batches
                batches = []
            batches.append(list(range(start_idx, start_idx + self.seq_len)))

        if len(batches) > 0 and not self.drop_last:
            yield batches

    def __len__(self):
        if self.drop_last:
            return (len(self.data_source) - self.seq_len + 1) // self.batch_size
        else:
            return (len(self.data_source) - self.seq_len + 1 + self.batch_size - 1) // self.batch_size
