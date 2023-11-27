import multiprocessing
import time
from zODfunction import OD

def script1():
  while True:
    print("SCRIPT1")
    time.sleep(1)
def script2():
  while True:
    print("SCRIPT2")
    time.sleep(1)
  
if __name__ == "__main__":
  # create two processes
  p1 = multiprocessing.Process(target=script1)
  p2 = multiprocessing.Process(target=script2)

  # start the processes
  p1.start()
  p2.start()

  # wait for the processes to finish
  p1.join()
  p2.join()
