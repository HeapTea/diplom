import os
import torch
import torch.nn as nn
from typing import List, Dict
from tsu_bot.ai.text_preprocessor import TextPreprocessor
from tsu_bot.ai.contants import CLASS_NAMES


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int, num_class: int):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text: int, offsets: List[float]):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


class Classifier:

    def __init__(self):
        self._model = torch.load(os.path.join(os.path.dirname(__file__), 'data/model.pth'))
        self._text_preprocessor = TextPreprocessor()

    def _model_predict(self, text: str) -> int:
        """
        :param text: текст
        :return: класс текста, определенный нейросетью
        """
        with torch.no_grad():
            text = torch.tensor(self._text_preprocessor.pipeline(text))
            output = self._model(text, torch.tensor([0]))
            return output.argmax(1).item()

    def _give_results(self, vector: Dict[int, float]) -> str:
        """
        :param vector: распределение классов групп
        :return: имя класса, или сообщение о том, что классификация не успешна
        """
        if sum(vector.values()) == 0:
            return 'Не получается определить факультет'
        class_ = max(vector, key=vector.get)
        result = CLASS_NAMES[class_]
        return result

    def clf(self, groups: List[str]):
        """
        :param groups: классификация профиля по группам
        :return: результа классификации
        """
        vector = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        for group in groups:
            group_class = self._model_predict(text=group)
            vector[group_class] += 1
        result = self._give_results(vector)
        return result

