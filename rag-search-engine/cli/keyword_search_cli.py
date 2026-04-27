#!/usr/bin/env python3

import argparse

from lib.keyword_search import (
    search_command,
    build_command,
    tf_command,
    idf_command,
    tf_idf_command,
    bm25_idf_command,
    bm25_tf_command,
    bm25search_command,
)

from lib.search_utils import (
    BM25_K1,
    BM25_B,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build and save the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Compute term frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to compute frequency for")

    idf_parser = subparsers.add_parser("idf", help="Compute inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to compute IDF for")

    tf_idf_parser = subparsers.add_parser("tfidf", help="Compute TF-IDF for a document and term")
    tf_idf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_idf_parser.add_argument("term", type=str, help="Term to compute TF-IDF for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to compute BM25 IDF for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to compute BM25 TF for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="BM25 k1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="BM25 b parameter")

    bm25_search_parser = subparsers.add_parser("bm25search", help="Search for documents using BM25")
    bm25_search_parser.add_argument("query", type=str, help="Query to search for")
    bm25_search_parser.add_argument("limit", type=int, nargs='?', default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for movie in results:
                print(f"{movie['title']} {movie['id']}")
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully")
        case "tf":
            print(tf_command(args.doc_id, args.term))
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tf_idf = tf_idf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            bm25_idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case "bm25tf":
            bm25_tf = bm25_tf_command(args.doc_id, args.term)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}")
        case "bm25search":
            results = bm25search_command(args.query, args.limit)
            for i, (doc_id, title, score) in enumerate(results):
                print(f"{i+1}. ({doc_id}) {title} - {score:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
