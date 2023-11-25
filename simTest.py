import multiprocessing
import time
from zODfunction import OD
if __name__ == "__main__":
  # create two processes
  p1 = multiprocessing.Process(target=OD)
  #p2 = multiprocessing.Process(target=script2)

  # start the processes
  p1.start()
  #p2.start()

  # wait for the processes to finish
  p1.join()
  #p2.join()
