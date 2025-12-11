import torch.nn as nn

class CharacterLM(nn.Module):
    """
    Character-level Language Model.
    Can be used for scoring candidate sequences during decoding.
    """
    def __init__(self, vocab_size=100, hidden_size=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def score(self, sequence):
        """
        Returns log probability of scalar sequence.
        """
        return 0.0
