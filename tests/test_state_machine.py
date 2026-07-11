# tests/test_state_machine.py
from companion.state_machine import StateMachine, State, Action


def test_starts_asleep():
    machine = StateMachine()
    assert machine.state == State.ASLEEP


def test_asleep_ignores_unrelated_text():
    machine = StateMachine()
    action = machine.process("what a nice day today")
    assert action == Action.IGNORE
    assert machine.state == State.ASLEEP


def test_asleep_wakes_on_wake_phrase():
    machine = StateMachine()
    action = machine.process("Hey Iris, how are you")
    assert action == Action.WAKE
    assert machine.state == State.ACTIVE


def test_wake_detection_is_case_insensitive():
    machine = StateMachine()
    action = machine.process("HEY IRIS")
    assert action == Action.WAKE


def test_wake_detection_survives_whisper_punctuation():
    # Whisper routinely returns "Hey, Iris!" — a raw substring check on
    # "hey iris" would miss it because of the comma.
    machine = StateMachine()
    action = machine.process("Hey, Iris!")
    assert action == Action.WAKE
    assert machine.state == State.ACTIVE


def test_asleep_wakes_on_misheard_wake_phrase():
    machine = StateMachine()
    action = machine.process("Hey Irish, can we practice?")
    assert action == Action.WAKE
    assert machine.state == State.ACTIVE


def test_stop_detection_survives_hyphenation():
    # Whisper routinely returns "Bye-bye!" for a spoken "bye bye".
    machine = StateMachine()
    machine.process("hey iris")
    action = machine.process("Bye-bye!")
    assert action == Action.SLEEP
    assert machine.state == State.ASLEEP


def test_cancel_detection_survives_punctuation():
    machine = StateMachine()
    machine.process("hey iris")
    action = machine.process("I went to the store, cancel that!")
    assert action == Action.CANCEL


def test_active_forwards_normal_speech():
    machine = StateMachine()
    machine.process("hey iris")
    action = machine.process("I want to practice grammar today")
    assert action == Action.FORWARD
    assert machine.state == State.ACTIVE


def test_active_cancels_on_cancel_phrase():
    machine = StateMachine()
    machine.process("hey iris")
    action = machine.process("I want to talk about, cancel that")
    assert action == Action.CANCEL
    assert machine.state == State.ACTIVE


def test_active_sleeps_on_stop_phrase():
    machine = StateMachine()
    machine.process("hey iris")
    action = machine.process("okay bye bye")
    assert action == Action.SLEEP
    assert machine.state == State.ASLEEP


def test_returns_to_asleep_behavior_after_sleeping():
    machine = StateMachine()
    machine.process("hey iris")
    machine.process("bye bye")
    action = machine.process("random chatter")
    assert action == Action.IGNORE
    action = machine.process("hey iris again please")
    assert action == Action.WAKE


def test_stop_takes_precedence_over_cancel_in_same_utterance():
    machine = StateMachine()
    machine.process("hey iris")
    action = machine.process("cancel that, bye bye")
    assert action == Action.SLEEP
    assert machine.state == State.ASLEEP
