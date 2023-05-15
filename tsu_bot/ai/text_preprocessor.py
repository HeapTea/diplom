import os
import pickle
from torchtext.data.utils import get_tokenizer


class TextPreprocessor:

    def __init__(self):
        filepath = os.path.join(os.path.dirname(__file__), 'data/vocab.pickle')
        self._vocab = pickle.load(open(filepath, 'rb'))
        self._tokenizer = get_tokenizer('basic_english')

    def _clear_text(self, text: str) -> str:
        """
        :param text: текст
        :return: текст без чисел и спецсимволов
        """
        cleared_text = str()
        for char in text.lower():
            if char.isalpha(): cleared_text += char
            else: cleared_text += ' '
        return cleared_text.strip()

    def pipeline(self, text):
        """
        :param text: текст
        :return: закодированный, с помощью словаря текст (вектор целых чисел)
        """
        text = self._clear_text(text)
        return self._vocab(self._tokenizer(text))

