from statistics import Statistics
import traceback

class Exit:
	def __init__(self):
		self.loop = True	
	def status(self):
		return self.loop
	def toggle(self):
		self.loop = not self.loop

def main():
	stats = Statistics('en_docs_clean.csv')
	e = Exit()
	options = {
		1 : stats.showMostMentionedEntitiesEachParty,
		2 : stats.showMostMentionedEntitiesGlobally,
		3 : stats.showMentionsMatrix,
		4 : stats.showMostMentionedParty,
		5 : stats.showHowManyTimesEachPartyMentionsOthers,
		6 : stats.showDistanceBetweenNamedEntities,
		7 : stats.showMostSimilarParties,
		8 : e.toggle
	}
	
	while(e.status()):
		print()
		print('Statistics:')
		print()
		print('1 - Show most mentioned entities for each party')
		print('2 - Show most mentioned entities globally')
		print('3 - Show which party mentions the other ones')
		print('4 - Show most mentioned party')
		print('5 - Show how many times each party mentions parties')
		print('6 - Show distance between named entities vectors')
		print('7 - Show similiarity scores between named entities vectors')
		print('8 - Exit')
		print()
		action=input('Display statistic: ')
		print()
		
		try:
			option = int(action)

			if 0 < option <= len(options):
				options[option]()
			else:
				print('Option should be between 1 and', len(options))
		except:
			print('Option should be between 1 and', len(options))
			traceback.print_exc()

if __name__ == "__main__":
	main()