import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def build_chain(model: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    return prompt | model | StrOutputParser()


def main() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GOOGLE_API_KEY in your .env file before starting the chat.")

    model = ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=api_key,
    )
    chain = build_chain(model)
    history: list[BaseMessage] = []

    print("Chat started. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("You: ").strip()
        except EOFError:
            print()
            break

        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        answer = chain.invoke({"history": history, "question": question})
        print(f"Assistant: {answer}")
        history.extend(
            [HumanMessage(content=question), AIMessage(content=answer)]
        )


if __name__ == "__main__":
    main()