from pseudoQuicksort import quicksort

FILE = "file.txt"

lst = []
with open(FILE) as file:
	for line in file:
		lst.append(int(line.replace('\n','')))

quicksort(lst)