import argparse

from lib.hybrid_search import normalize_scores, rrf_search_command, weighted_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument(
        "scores", nargs="+", type=float, help="List of scores to normalize"
    )

    weighted_parser = subparsers.add_parser(
        "weighted-search", help="Perform weighted hybrid search"
    )
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for BM25 vs semantic (0=all semantic, 1=all BM25, default=0.5)",
    )
    weighted_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    rrf_parser = subparsers.add_parser(
        "rrf-search", help="Perform RRF hybrid search"
    )
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument(
        "--k", type=int, default=60, help="RRF constant (default=60)"
    )
    rrf_parser.add_argument(
        "--limit", type=int, default=10, help="Number of results to return (default=10)"
    )
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            result = weighted_search_command(args.query, args.alpha, args.limit)

            print(
                f"Weighted Hybrid Search Results for '{result['query']}' (alpha={result['alpha']}):"
            )
            print(
                f"  Alpha {result['alpha']}: {int(result['alpha'] * 100)}% Keyword, {int((1 - result['alpha']) * 100)}% Semantic"
            )
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(
                        f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
                    )
                print(f"   {res['document'][:100]}...")
                print()
        case "rrf-search":
            result = rrf_search_command(args.query, args.k, args.limit, args.enhance)

            print(
                f"RRF Hybrid Search Results for '{result['query']}' (k={result['k']}):"
            )
            print()

            if result["enhanced_query"]:
                print(f"Enhanced query: ({result['enhanced_method']}): {result["query"]} -> {result['enhanced_query']}\n")

            for i, res in enumerate(result["results"], 1):
                metadata = res.get("metadata", {})
                bm25_rank = metadata.get("bm25_rank")
                semantic_rank = metadata.get("semantic_rank")
                bm25_str = bm25_rank if bm25_rank is not None else "-"
                semantic_str = semantic_rank if semantic_rank is not None else "-"
                print(f"{i}. {res['title']}")
                print(f"  RRF Score: {res['score']:.3f}")
                print(f"  BM25 Rank: {bm25_str}, Semantic Rank: {semantic_str}")
                print(f"  {res['document'][:100]}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
