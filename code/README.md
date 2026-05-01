# Code Folder Quickstart

This folder contains a terminal-based Python agent that reads `support_tickets/support_tickets.csv` and writes `support_tickets/output.csv`.

## Setup

1. Create the virtual environment from the repo root:

```powershell
C:/Program Files/Python314/python.exe -m venv .venv
```

2. Run the agent:

```powershell
.\.venv\Scripts\python.exe code\main.py --input support_tickets\support_tickets.csv --output support_tickets\output.csv
```

## Notes

- The agent uses only the markdown corpus in `data/`.
- No external API keys are required.
- The implementation is deterministic and purely local.