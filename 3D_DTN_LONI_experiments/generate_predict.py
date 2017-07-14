import os
for i in range(2,21):
    print("generate predict file",i)
    os.system("cp main.py main"+str(i)+".py")
