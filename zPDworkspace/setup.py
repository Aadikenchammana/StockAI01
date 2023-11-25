import os
import subprocess

print('h')
# Change current working directory to 'yolov7'
#os.chdir('yolov7')
print("chdir")

# Install dependencies using pip
subprocess.run(['pip3', 'install', '-qr', 'requirements.txt'])
print("run")
