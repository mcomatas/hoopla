import argparse

from lib.augmented_generation import (
    rag_command,
    summarize_command,
    citations_command,
    question_command
)

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize search results"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for summarization")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Number of search results to use")

    citations_parser = subparsers.add_parser(
        "citations", help="Generate citations for search results"
    )
    citations_parser.add_argument("query", type=str, help="Search query for citations")
    citations_parser.add_argument("--limit", type=int, default=5, help="Number of search results to use")

    question_parser = subparsers.add_parser(
        "question", help="Ask a question and get an answer"
    )
    question_parser.add_argument("question", type=str, help="Question to ask")
    question_parser.add_argument("--limit", type=int, default=5, help="Number of search results to use")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            results = rag_command(query)

            print("Search Results:")
            for search_result in results[0]:
                print("- ", search_result['title'])

            print("RAG Response: ")
            print(results[1].text)
        case "summarize":
            query = args.query
            results = summarize_command(query)

            print("Search Results:")
            for search_result in results[0]:
                print("- ", search_result['title'])

            print("Summary Response: ")
            print(results[1].text)
        case "citations":
            query = args.query
            results = citations_command(query)

            print("Search Results:")
            for search_result in results[0]:
                print("- ", search_result['title'])

            print("Citations Response: ")
            print(results[1].text)
        case "question":
            question = args.question
            results = question_command(question, args.limit)

            print("Search Results:")
            for search_result in results[0]:
                print("- ", search_result['title'])

            print("Question Response: ")
            print(results[1].text)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
