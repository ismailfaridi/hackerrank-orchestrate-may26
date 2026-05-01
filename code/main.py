from __future__ import annotations

import argparse
import csv
from pathlib import Path

from agent import SupportTicketAgent


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run the support triage agent")
	parser.add_argument(
		"--input",
		default="support_tickets/support_tickets.csv",
		help="Path to the input CSV file.",
	)
	parser.add_argument(
		"--output",
		default="support_tickets/output.csv",
		help="Path to write the predictions CSV.",
	)
	parser.add_argument(
		"--data-root",
		default="data",
		help="Root folder containing the support corpus.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	repo_root = Path(__file__).resolve().parent.parent
	data_root = (repo_root / args.data_root).resolve()
	input_path = (repo_root / args.input).resolve()
	output_path = (repo_root / args.output).resolve()

	agent = SupportTicketAgent(data_root=data_root)

	with input_path.open("r", encoding="utf-8", newline="") as input_file:
		rows = list(csv.DictReader(input_file))

	predictions = [agent.triage_row(row) for row in rows]

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8", newline="") as output_file:
		writer = csv.DictWriter(
			output_file,
			fieldnames=["status", "product_area", "response", "justification", "request_type"],
		)
		writer.writeheader()
		writer.writerows(predictions)


if __name__ == "__main__":
	main()
