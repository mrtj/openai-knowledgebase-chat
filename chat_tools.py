import enum

class Language(enum.Enum):
    EN = 'en'
    IT = 'it'

class MessageType(enum.Enum):
    SYSTEM: str = 'system'
    USER: str = 'user'
    ASSISTANT: str = 'assistant'
