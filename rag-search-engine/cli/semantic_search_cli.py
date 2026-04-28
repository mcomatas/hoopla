#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    semantic_search,
    chunk,
    semantic_chunking,
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the semantic search model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed text using the semantic search model")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify the embeddings")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embed query text using the semantic search model")
    embed_query_parser.add_argument("query", type=str, help="Query text to embed")

    search_parser = subparsers.add_parser("search", help="Perform a semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Size of each chunk")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap between chunks")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk text into smaller pieces using semantic similarity")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Size of each chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap between chunks")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            semantic_search(args.query, args.limit)
        case "chunk":
            chunk(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunking(args.text, args.max_chunk_size, args.overlap)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
