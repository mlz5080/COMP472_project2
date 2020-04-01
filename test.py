import subprocess

if __name__ == '__main__':
	for j in range(1,4):
		for i in range(3):
			print('python3', 'project2.py',str(i),str(j),str(0.1))
			subprocess.call(['python3', 'project2.py',str(i),str(j),str(0.1)])