import argparse
from lib.evaluation import evaluate_golden_dataset

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    results = evaluate_golden_dataset(limit)
    print(f"k={limit}\n")
    for query, data in results.items():
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {data['precision']:.4f}")
        print(f"  - Retrieved: {', '.join(data['retrieved'])}")
        print(f"  - Relevant: {', '.join(data['relevant'])}")
        print()

if __name__ == "__main__":
    main()
