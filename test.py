import subprocess

if __name__ == '__main__':

	"""
	1. your BYOM
	2. the required model with V=0 n=1 d=0
	3. the required model with V=1 n=2 d=0.5
	4. the required model with V=1 n=3 d=1
	5. the required model with V=2 n=2 d=0.3
	"""

	
	
	print('python3', 'project2.py',"./OriginalDataSet/training-tweets.txt","./demo/test12.txt")	
	print("BYOM")	
	subprocess.call(['python3', 'project2.py',"./OriginalDataSet/training-tweets.txt","./demo/test12.txt"])
	
	print('python3', 'project2.py',str(0),str(1),str(0),"./OriginalDataSet/training-tweets.txt","./demo/test12.txt")
	subprocess.call(['python3', 'project2.py',str(0),str(1),str(0),"./OriginalDataSet/training-tweets.txt","./demo/test12.txt"])

	print('python3', 'project2.py',str(1),str(2),str(0.5),"./OriginalDataSet/training-tweets.txt","./demo/test12.txt")
	subprocess.call(['python3', 'project2.py',str(1),str(2),str(0.5),"./OriginalDataSet/training-tweets.txt","./demo/test12.txt"])

	print('python3', 'project2.py',str(1),str(3),str(1),"./OriginalDataSet/training-tweets.txt","./demo/test12.txt")
	subprocess.call(['python3', 'project2.py',str(1),str(3),str(1),"./OriginalDataSet/training-tweets.txt","./demo/test12.txt"])

	print('python3', 'project2.py',str(2),str(2),str(0.3),"./OriginalDataSet/training-tweets.txt","./demo/test12.txt")
	subprocess.call(['python3', 'project2.py',str(2),str(2),str(0.3),"./OriginalDataSet/training-tweets.txt","./demo/test12.txt"])
	
	
		#,1,0.1,0.5,0.09
	"""
	smooth_values = [0,1,0.1,0.5,0.09]
	V=[1]
	size = [2]
	for s in size:
		for v in V:
			for sm in smooth_values:
				print('python3', 'project2.py',str(v),str(s),str(sm),"./OriginalDataSet/training-tweets.txt","./OriginalDataSet/test-tweets-given.txt")
				subprocess.call(['python3', 'project2.py',str(v),str(s),str(sm),"./OriginalDataSet/training-tweets.txt","./OriginalDataSet/test-tweets-given.txt"])
	"""

	"""
	for s in size:
		for v in V:
			for sm in smooth_values:
				print('python3', 'project2.py',str(v),str(s),str(sm),"./OriginalDataSet/training-tweets.txt","./demo/test12.txt")
				subprocess.call(['python3', 'project2.py',str(v),str(s),str(sm),"./OriginalDataSet/training-tweets.txt","./demo/test12.txt"])
	"""
