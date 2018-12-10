import codecs

class DataManager:
    def __init__(self):
        print('SemanticAnalogies data manager initialized')

    @staticmethod
    def create_vocab(vectors):
        words = vectors['name']
        vocab_size = len(words)
        vocab = {w: idx for idx, w in enumerate(words)}

        return vocab

    @staticmethod
    def read_data(vocab, gold_standard_file):
        full_data = []
        file_input_stream = codecs.open(gold_standard_file, 'r', 'utf-8')
        for line in file_input_stream:
            full_data.append(line.rstrip().split())

        data = [x for x in full_data if all(word in vocab for word in x)]
        
        ignored = [x for x in full_data if not x in data]
        return data, ignored
