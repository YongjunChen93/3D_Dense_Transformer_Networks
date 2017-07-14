import os
gpu = 4
os.system("CUDA_VISIBLE_DEVICES="+str(gpu)+" python main.py --action=predict")
for i in range(2,21):
    os.system("CUDA_VISIBLE_DEVICES="+str(gpu)+" python main"+str(i)+".py --action=predict")
