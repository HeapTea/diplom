import vk_api
from typing import List
from tsu_bot.bot.config import *


class GroupUsersCollector:

    def __init__(self):
        self._vk_session = vk_api.VkApi(LOGIN, PASSWORD)
        self._vk_session.auth()
        self._vk = self._vk_session.get_api()

    def collect(self, user_id: str) -> List[str]:
        """
        :param user_id: id пользователя vk
        :return: список групп пользователя
        """
        groups = self._vk.groups.get(user_id=user_id, extended=1, fields='name')['items']
        groups = [x['name'] for x in groups]
        return groups

