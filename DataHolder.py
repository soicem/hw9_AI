import numpy as np


class DataHolder:
    def __init__(self):
        words_file = open('word_dic', 'r', encoding='utf-8')
        self.Words = np.array(words_file.read().split('\n'), dtype='<U20')
        self.Words.sort()

        path = 'train_processed.txt'

        train_file = open(path, 'r', encoding='utf-8')
        lines = train_file.read().split('\n')

        max_length = 0

        for i in range(len(lines)):
            TK = lines[i].split(' \t')

            words = TK[0].split(' ')

            if max_length < len(words):
                max_length = len(words)

        print(max_length)

        self.input_ids = np.zeros(shape=[len(lines), max_length], dtype=np.int32)
        self.label = np.zeros(shape=[len(lines)], dtype=np.int32)

        for i in range(len(lines) - 1):
            TK = lines[i].split(' \t')
            if len(TK) != 2:
                TK = lines[i].split('\t')

            words = TK[0].split(' ')
            for idx, j in enumerate(range(max_length - len(words), max_length)):
                self.input_ids[i, j] = self.Words.searchsorted(words[idx])

            self.label[i] = int(TK[1])

        path = 'test_processed.txt'

        test_file = open(path, 'r', encoding='utf-8')
        lines = test_file.read().split('\n')

        max_length = 0

        for i in range(len(lines)):
            TK = lines[i].split(' \t')

            words = TK[0].split(' ')

            if max_length < len(words):
                max_length = len(words)

        print(max_length)

        self.test_input_ids = np.zeros(shape=[len(lines), max_length], dtype=np.int32)
        self.test_label = np.zeros(shape=[len(lines)], dtype=np.int32)

        for i in range(len(lines) - 1):
            TK = lines[i].split(' \t')
            if len(TK) != 2:
                TK = lines[i].split('\t')

            words = TK[0].split(' ')
            for idx, j in enumerate(range(max_length - len(words), max_length)):
                self.test_input_ids[i, j] = self.Words.searchsorted(words[idx])

            self.test_label[i] = int(TK[1])

        self.Batch_Size = 256

        self.random_idx = np.array(range(self.label.shape[0]), dtype=np.int32)
        np.random.shuffle(self.random_idx)

        self.Batch_Idx = 0
        self.Test_Batch_Idx = 0

    def next_random_batch(self):
        if self.Batch_Idx >= 80000:
            np.random.shuffle(self.random_idx)
            self.Batch_Idx = 0

        x = np.zeros(shape=[self.Batch_Size, self.input_ids.shape[1]], dtype=np.int32)
        Y = np.zeros(shape=[self.Batch_Size, 2], dtype=np.float32)

        for i in range(self.Batch_Size):
            idx = self.random_idx[self.Batch_Idx]

            x[i] = self.input_ids[idx]
            Y[i, self.label[idx]] = 1

            self.Batch_Idx += 1

        return x, Y

    def next_test_batch(self):
        if self.Test_Batch_Idx + self.Batch_Size >= self.label.shape[0]:
            return False, 0, 0

        x = np.zeros(shape=[self.Batch_Size, self.test_input_ids.shape[1]], dtype=np.int32)
        Y = np.zeros(shape=[self.Batch_Size], dtype=np.float32)

        for i in range(self.Batch_Size):
            idx = self.Test_Batch_Idx

            x[i] = self.test_input_ids[idx]
            Y[i] = self.test_label[idx]

            self.Test_Batch_Idx += 1

        return True, x, Y

