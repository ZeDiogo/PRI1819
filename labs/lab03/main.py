from Data import Data

def main():
	data = Data()
	data.createVectors()
	data.createClassifier()
	data.printResults()
	data.kmeansClustering()
	

if __name__ == "__main__":
	main()