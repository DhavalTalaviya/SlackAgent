import argparse

from src.agent.qa_graph import ask, qa_session, run_turn


def _print_result(result: dict) -> None:
    print("\n--- Answer ---")
    print(result["answer"])
    if result["citations"]:
        print("\n--- Sources ---")
        for c in result["citations"]:
            print(f"[{c['marker']}] #{c['channel']} - {c['user']} - {c['permalink']}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask the Slack knowledge agent a question.")
    parser.add_argument(
        "--user",
        required=True,
        help="Slack user ID to ask as (used for permission-aware retrieval, e.g. U01ABC123).",
    )
    parser.add_argument("question", nargs="*", help="The question. Omit for interactive mode.")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    thread_id = f"user:{args.user}"

    if question:
        _print_result(ask(question, requesting_user_id=args.user, thread_id=thread_id))
    else:
        print(f"Interactive mode as {args.user}. Follow-up questions remember the conversation. Ctrl+C to quit.\n")
        with qa_session() as graph:
            while True:
                try:
                    question = input("Question: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not question:
                    continue
                _print_result(run_turn(graph, question, args.user, thread_id))
