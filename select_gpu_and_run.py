import nvgpu
import subprocess
import os 
import sys

#usage: python2.7 program task 
task = sys.argv[1]

gpu_list = nvgpu.available_gpus()
s = './run.sh ' + str(gpu_list[0]) + ' ' + task
os.system(s)
#print s
#subprocess.call([s])

