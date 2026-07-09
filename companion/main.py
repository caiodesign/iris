# companion/main.py
import argparse
import sys

from dotenv import load_dotenv

from companion import config, session
from companion.session import PROVIDER_NAMES, STT_NAMES


def choose_from_menu(label, names, default, cli_choice) -> str:
    if cli_choice:
        return cli_choice
    print(f"Choose {label}:")
    for number, name in enumerate(names, start=1):
        marker = " (default)" if name == default else ""
        print(f"  {number}. {name}{marker}")
    answer = input("Number or name [Enter = default]: ").strip().lower()
    if not answer:
        return default
    if answer.isdigit() and 1 <= int(answer) <= len(names):
        return names[int(answer) - 1]
    if answer in names:
        return answer
    print(f"Unknown choice '{answer}', using {default}.")
    return default


def ask_yes_no(label, default, cli_flag) -> bool:
    if cli_flag:
        return True
    answer = input(f"{label} [y/N]: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def print_event(event) -> None:
    kind = event["event"]
    if kind == "heard":
        print(f"Heard: {event['text']}")
    elif kind == "reply":
        print(f"Companion: {event['text']}")
    elif kind == "system":
        print(event["text"])
    elif kind == "warning":
        print(f"WARNING: {event['text']}")
    elif kind == "error":
        print(f"ERROR: {event['text']}")
    # "status" and "session_ended" drive the web page; the terminal skips them.


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Voice English companion")
    parser.add_argument(
        "--brain",
        choices=PROVIDER_NAMES,
        default=None,
        help="skip the startup menu and use this provider",
    )
    parser.add_argument(
        "--ears",
        choices=STT_NAMES,
        default=None,
        help="skip the ears menu and use this transcription backend",
    )
    parser.add_argument(
        "--ptt",
        action="store_true",
        help="enable push-to-talk (hold PTT_KEY to record) and skip the prompt",
    )
    args = parser.parse_args()
    brain = choose_from_menu("brain", PROVIDER_NAMES, config.LLM_PROVIDER, args.brain)
    print(f"Brain: {brain}")
    ears = choose_from_menu(
        "ears (transcription)", STT_NAMES, config.STT_PROVIDER, args.ears
    )
    print(f"Ears: {ears}")
    ptt = ask_yes_no("Push-to-talk (hold a key to record)?", False, args.ptt)
    print(f"Push-to-talk: {'on' if ptt else 'off'}")

    print("Checking services and microphone...")
    try:
        ok = session.run_session(brain, ears, ptt, print_event, lambda: False)
    except KeyboardInterrupt:
        print("\nExiting.")
        return
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
