import time
import sys

class Bar:
	
	def __init__(self, sampleSize, toolbarSize=40, timer=False):
		self.toolbarSize = toolbarSize
		self.toolbarInc = 0
		self.sampleSize = sampleSize
		self.sampleInc = 0
		self.percentage = 0
		self.timer = timer
		if self.timer:
			self.start = time.time()
		sys.stdout.write("[%s] - %d" % ((" " * self.toolbarSize), 0))
		sys.stdout.write("%")
		sys.stdout.flush()

	def updateMapping(self):
		self.percentage = int(self.sampleInc * 100 / self.sampleSize)
		self.toolbarInc = int(self.sampleInc * self.toolbarSize / self.sampleSize)

	def update(self, inc=1):
		self.sampleInc += inc
		self.updateMapping()
		sys.stdout.write("\b" * (self.toolbarSize+1+20))
		sys.stdout.write("[%s] - %d" % ((" " * self.toolbarSize), self.percentage))
		sys.stdout.write("%")
		if self.timer:
			elapsed = int(time.time() - self.start)
			sys.stdout.write(" (%dsec)" % (elapsed))
			sys.stdout.write("\b" * (self.toolbarSize+11+len(str(self.percentage))+len(str(elapsed))))
		else:
			sys.stdout.write("\b" * (self.toolbarSize+5+len(str(self.percentage))))
		for i in range(self.toolbarInc):
			sys.stdout.write("-")
		sys.stdout.flush()
		if self.sampleInc >= self.sampleSize:
			sys.stdout.write("\n")
			sys.stdout.flush()

#Example of usage:
# bar = Bar(10, timer=True)
# for i in range(10):
# 	time.sleep(0.1)
# 	bar.update()
	