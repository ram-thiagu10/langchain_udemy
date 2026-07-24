from dotenv import load_dotenv
import os
load_dotenv()

def main():
    print("Hello from react-agent-langgraph!")
    print(os.environ.get("LANGSMITH_PROJECT"))


if __name__ == "__main__":
    main()
