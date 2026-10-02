from app.agent.core import ask_agent
from app.rag import index_repository


def run():
    print("Talk2Code: talk to your codebase\n")

    url = input("GitHub repo URL to index (press Enter to skip if already indexed): ").strip()
    if url:
        try:
            index_repository(url)
        except Exception as e:
            print(f"\nIndexing failed: {e}")
            return

    print("\nAsk a question about the repo (type 'exit' to quit)\n")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        try:
            answer = ask_agent(question)
        except Exception as e:
            print(f"\nError: {e}\n")
            continue
        print(f"\nTalk2Code: {answer}\n")


if __name__ == "__main__":
    run()   