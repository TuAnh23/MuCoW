import spacy
from spacy.tokens import Doc, Token


class WhitespaceTokenizer:
    def __init__(self, nlp):
        self.vocab = nlp.vocab

    def __call__(self, text):
        words = text.split()
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)
