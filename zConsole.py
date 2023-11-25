import multiprocessing
import time
import json
from zDCfunction import DC
from zODfunction import OD
from zPDfunction import PD
if __name__ == "__main__":
    # create two processes
    p1 = multiprocessing.Process(target=DC)
    p2 = multiprocessing.Process(target=OD)
    p3 = multiprocessing.Process(target=PD)


    with open('zODworkspace//watchlist.txt', 'w') as f:
        json.dump({}, f)
    with open('zODworkspace//total_watchlist.txt', 'w') as f:
        json.dump({}, f)

    p1.start()
    time.sleep(1)
    p2.start()
    time.sleep(0.5)
    p3.start()

    # wait for the processes to finish
    p1.join()
    p2.join()
    p3.join()
