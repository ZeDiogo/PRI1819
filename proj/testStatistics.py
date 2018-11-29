from statistics import Statistics

def main():
	stats = Statistics('en_docs_clean.csv')
	stats.showMostMentionedEntitiesEachParty(top=3)
	stats.showMostMentionedEntitiesGlobally()

if __name__ == "__main__":
	main()