"""Фазы диалога и тип пользователя."""


class UserState:
    WHO = "who"
    ABOUT = "about"
    TEST = "test"
    RECOMMENDATION = "recommendation"
    TALK = "talk"
    INJECT_ATTEMPT = "inject_attempt"


class UserType:
    SCHOOL = "school"
    STUDENT = "student"
    WORKER = "worker"
