"""
ASPI LLM Judge — CLI
====================
Batch-annotate saved evaluation results with LLM-as-judge classifications.

Usage:
    # Judge all groups
    python run_judge.py results.jsonl -o results_judged.jsonl

    # Judge first 10 groups only
    python run_judge.py results.jsonl -o results_judged.jsonl --limit 10

    # Use a specific model
    python run_judge.py results.jsonl --model openai/gpt-4o-mini-2024-07-18

    # Use OpenAI backend directly (instead of LiteLLM proxy)
    python run_judge.py results.jsonl --backend openai
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aspi_judge import ASPIJudge


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-as-judge on saved ASPI evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_judge.py output_analysis/results.jsonl
  python run_judge.py results.jsonl -o results_judged.jsonl --limit 10
  python run_judge.py results.jsonl --model openai/gpt-4o-mini-2024-07-18 --backend openai
        """,
    )
    parser.add_argument("input", help="Path to evaluation results JSONL")
    parser.add_argument("-o", "--output", help="Output path (default: input with _judged suffix)")
    parser.add_argument("--model", default="openai/gpt-4o-mini-2024-07-18",
                        help="Judge model (default: openai/gpt-4o-mini-2024-07-18)")
    parser.add_argument("--backend", default="litellm", choices=["litellm", "openai"],
                        help="Backend: litellm (proxy) or openai (direct)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max groups to judge (0 = all)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path).replace(".jsonl", "_judged.jsonl")
        if output_path == str(input_path):
            output_path = str(input_path) + ".judged"

    print(f"Input:   {input_path}")
    print(f"Output:  {output_path}")
    print(f"Model:   {args.model}")
    print(f"Backend: {args.backend}")
    if args.limit:
        print(f"Limit:   {args.limit}")
    print()

    judge = ASPIJudge(model=args.model, backend=args.backend)
    judge.run_judge_pass(str(input_path), output_path, limit=args.limit)


if __name__ == "__main__":
    main()
