class AccessError(Exception):

    def __str__(self):
        return """Доступ к профилю закрыт. Причины: 
        1. Закрытый профиль ВК
        2. Закрыт список групп в настройках приватности
        """