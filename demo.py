from TrainDetector import TrainDetector
from option import Options
import sys

if __name__ == '__main__':
    detector = TrainDetector()
    opt = Options()
    if len(sys.argv) != 2:
        print("NPD\n \
               test:   test one image\n \
               live:   live demo with camera support\n")
    elif sys.argv[1] == 'test':
        detector.Detect(opt)
    elif sys.argv[1] == 'live':
        detector.Live(opt)
