DEBUG = False

def quicksort(lst):
	quicksortBase(lst, 0, len(lst)-1)
	print(lst)

def quicksortBase(lst, low, high):
	if low < high:
		pivot_index = 0;
		pivot_index = partition(lst, low, high)
		quicksortBase(lst, low, pivot_index-1)
		quicksortBase(lst, pivot_index+1, high)

def swap(lst, i, j):
	aux = lst[i]
	lst[i] = lst[j]
	lst[j] = aux
	if DEBUG: print(lst, lst[i], ' <--> ', lst[j])

def partition(lst, low, high):
	pivot_index = low
	for i in range(low+1, high+1):
		if lst[i] <= lst[low]:
			pivot_index += 1
			swap(lst, i, pivot_index)
	swap(lst, low, pivot_index)
	return pivot_index

# def main():
# 	#lst = [2,7,9,1,12,5,0,3]
# 	lst = [0,4353,23,9,12,1,1,1,3,1,-99,-150,0,2,7,1]
# 	print(lst, ' <-- Initial')
# 	quicksort(lst)

# if __name__ == "__main__":
# 	main()
