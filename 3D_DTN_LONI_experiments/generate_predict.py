import os
import re

file_path = './main.py'
for i in range(2,21):
	f = open(file_path,"r+")
	print("generate_file:",i)
	open('main'+str(i)+'.py', 'w').write(re.sub('index = 1', 'index = '+str(i), f.read()))
	f.close()