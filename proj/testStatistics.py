from statistics import Statistics
import traceback

def main():
	stats = Statistics('en_docs_clean.csv')
	options = {
		1 : stats.showMostMentionedEntitiesEachParty,
		2 : stats.showMostMentionedEntitiesGlobally,
		3 : stats.showMostMentionedPartyByOthers,
	}
	while(True):
		print()
		print('Statistics:')
		print()
		print('1 - Show most mentioned entities for each party')
		print('2 - Show most mentioned entities globally')
		print('3 - Show which party mentiones the other ones')
		# print('4 - ')
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


		print()
		exit=input('Press enter to continue or write \"exit\" to stop: ')
		if exit == 'exit' or exit == 'quit' or exit == 'stop':
			break

if __name__ == "__main__":
	main()