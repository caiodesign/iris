# companion/state_machine.py
import re
from enum import Enum, auto

from companion.config import CANCEL_PHRASE, STOP_PHRASE, WAKE_PHRASE


class State(Enum):
    ASLEEP = auto()
    ACTIVE = auto()


class Action(Enum):
    IGNORE = auto()
    WAKE = auto()
    SLEEP = auto()
    CANCEL = auto()
    FORWARD = auto()


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation so Whisper's "Hey, Chat!" or
    "Bye-bye!" still match the plain control phrases."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class StateMachine:
    def __init__(self):
        self.state = State.ASLEEP

    def process(self, text: str) -> Action:
        normalized = _normalize(text)

        if self.state == State.ASLEEP:
            if WAKE_PHRASE in normalized:
                self.state = State.ACTIVE
                return Action.WAKE
            return Action.IGNORE

        if STOP_PHRASE in normalized:
            self.state = State.ASLEEP
            return Action.SLEEP
        if CANCEL_PHRASE in normalized:
            return Action.CANCEL
        return Action.FORWARD
