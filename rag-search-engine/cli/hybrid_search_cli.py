import argparse
import logging
import os

from lib.hybrid_search import (
    normalize_scores,
    rrf_search_command,
    weighted_search_command,
    evaluate_results,
)

if os.getenv("DEBUG"):
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s %(name)s] %(message)s")
    logging.getLogger("lib").setLevel(logging.DEBUG)


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
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Rerank the top documents"
    )

    rrf_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the search results against a ground truth file",
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
            result = rrf_search_command(args.query, args.k, args.limit, args.enhance, args.rerank_method)

            print(
                f"RRF Hybrid Search Results for '{result['query']}' (k={result['k']}):"
            )
            print()

            if result["enhanced_query"]:
                print(f"Enhanced query: ({result['enhanced_method']}): {result["query"]} -> {result['enhanced_query']}\n")

            if result["reranked"]:
                print(f"Re-ranking top {len(result['results'])} results using {result['rerank_method']}...\n")
                print()

            for i, res in enumerate(result["results"], 1):
                metadata = res.get("metadata", {})
                bm25_rank = metadata.get("bm25_rank")
                semantic_rank = metadata.get("semantic_rank")
                bm25_str = bm25_rank if bm25_rank is not None else "-"
                semantic_str = semantic_rank if semantic_rank is not None else "-"
                print(f"{i}. {res['title']}")
                if "individual_score" in res:
                    print(f"  Re-rank Score: {res.get("individual_score", 0):.3f}/10")
                if "batch_rank" in res:
                    print(f"   Re-rank Rank: {res.get('batch_rank', 0)}")
                if "cross_encoder_score" in res:
                    print(f"  Cross-Encoder Score: {res['cross_encoder_score']:.3f}")
                print(f"  RRF Score: {res['score']:.3f}")
                print(f"  BM25 Rank: {bm25_str}, Semantic Rank: {semantic_str}")
                print(f"  {res['document'][:100]}...")
                print()

            if args.evaluate:
                evaluated_results = evaluate_results(result["query"], result["results"])
                for i, res in enumerate(evaluated_results, start=1):
                    print(f"{i}. {res['title']}: {res['relevance_score']}/3")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
