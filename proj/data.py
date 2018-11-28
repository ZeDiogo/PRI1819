import pandas as pd

class Data:

	def __init__(self, filename):
		self.data = pd.read_csv(filename, sep=',', header=0)
		# self.data.columns = ["text","manifesto_id", "party", "date", "title"]
		# print(self.data)

	def getTexts(self):
		return self.data["text"].tolist()

	def getManifestoIds(self):
		return self.data["manifesto_id"].tolist()

	def getParties(self):
		return self.data["party"].tolist()

	def getDates(self):
		return self.data["date"].tolist()

	def getTitles(self):
		return self.data["title"].tolist()

	def getPartiesTexts(self):
		parties = self.getParties()
		texts = self.getTexts()
		# print(parties)
		# for t in self.getTexts():
		# 	print(t, '\n\n\n\n\n\n\n\n')
		return [(parties[i], text) for i, text in enumerate(texts)]
		# print(aux)
		# return 