import subprocess

if __name__ == '__main__':

	"""
	1. your BYOM
	2. the required model with V=0 n=1 d=0
	3. the required model with V=1 n=2 d=0.5
	4. the required model with V=1 n=3 d=1
	5. the required model with V=2 n=2 d=0.3
	"""
	print('python3', 'project2.py')	
	print("BYOM")	
	subprocess.call(['python3', 'project2.py'])

	print('python3', 'project2.py',str(0),str(1),str(0))
	subprocess.call(['python3', 'project2.py',str(0),str(1),str(0)])

	print('python3', 'project2.py',str(1),str(2),str(0.5))
	subprocess.call(['python3', 'project2.py',str(1),str(2),str(0.5)])

	print('python3', 'project2.py',str(1),str(3),str(1))
	subprocess.call(['python3', 'project2.py',str(1),str(3),str(1)])

	print('python3', 'project2.py',str(2),str(2),str(0.3))
	subprocess.call(['python3', 'project2.py',str(2),str(2),str(0.3)])

	