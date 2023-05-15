import random
from typing import List
from vk_api.exceptions import ApiError
from vk_api.longpoll import VkLongPoll, VkEventType
from tsu_bot.bot.config import *
from tsu_bot.bot.utils import *
from tsu_bot.bot.errors import AccessError
from tsu_bot.ai.models import *


class Bot:

    def __init__(self):
        vk_session = vk_api.VkApi(token=ACCESS_TOKEN)
        self._long_poll = VkLongPoll(vk_session)
        self._vk = vk_session.get_api()
        self._collector = GroupUsersCollector()
        self._classifier = Classifier()

    def _send_message(self, user_id: str, message: str):
        """
        :param user_id: id пользователя vk
        :param message: сообщение
        """
        self._vk.messages.send(user_id=user_id, message=message, random_id=random.randint(1, 1000))

    def _get_groups(self, user_id: str) -> List[str]:
        """
        :param user_id: id пользователя vk
        :return: список групп пользователя
        """
        try:
            groups = self._collector.collect(user_id)
            return groups
        except ApiError:
            raise AccessError

    def start(self):
        for event in self._long_poll.listen():
            if event.type == VkEventType.MESSAGE_NEW and event.to_me:
                request = event.text.lower()
                user_id = event.user_id
                if request == '/факультет':
                    groups = self._get_groups(user_id)
                    message = self._classifier.clf(groups)
                    self._send_message(user_id, message)

