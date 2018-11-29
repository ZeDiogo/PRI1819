from statistics import Statistics

def main():
	stats = Statistics('en_docs_clean.csv')
	stats.showMostMentionedEntitiesEachParty(top=3)
	stats.showMostMentionedEntitiesGlobally(minMentions=0)
	stats.showMostMentionedPartyByOthers()
	
if __name__ == "__main__":
	main()