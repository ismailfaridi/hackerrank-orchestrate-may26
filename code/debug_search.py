from pathlib import Path
from corpus import CorpusIndex

ci = CorpusIndex(Path("..") / "data")
query = "My mock interviews stopped in between. What should i do now?"
print('Query:', query)
print('\nCompany-scoped results (company=Claude):')
for art, score in ci.search(query=query, company='Claude', limit=6):
    print(f"{score:.2f}	{art.relative_path}")
print('\nUnscoped results:')
for art, score in ci.search(query=query, company=None, limit=6):
    print(f"{score:.2f}	{art.relative_path}")
print('\nDiagnostics: list articles with "interview" in path or title')
count = 0
for a in ci.articles:
    if 'interview' in a.relative_path or 'interview' in a.title.lower():
        print('MATCH', a.relative_path)
        count += 1
        if count > 50:
            break
print('\nFound', count, 'articles with "interview" in path/title')
