from pathlib import Path
from agent import SupportTicketAgent

repo_root = Path(__file__).resolve().parent.parent
agent = SupportTicketAgent(repo_root / 'data')
issue = "My mock interviews stopped in between. What should i do now?"
subject = "Information"
company = "Claude"
print('Issue:', issue)
print('Subject:', subject)
print('Company:', company)
decision = agent.triage(issue=issue, subject=subject, company=company)
print('\nDecision:')
print('status:', decision.status)
print('product_area:', decision.product_area)
print('request_type:', decision.request_type)
print('justification:', decision.justification)
print('\nResponse:\n', decision.response)
