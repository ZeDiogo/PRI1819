from statistics import Statistics

def main():
	stats = Statistics('en_docs_clean.csv')
	stats.getMostMentionedEntities('United Kingdom Independence Party')
	stats.getMostMentionedEntities('Conservative Party')
	stats.getMostMentionedEntities('Labour Party')

if __name__ == "__main__":
	main()