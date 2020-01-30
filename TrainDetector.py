from LearnGAB import GAB
import time
import cv2

class TrainDetector:
    def Detect(self, opt):
        """
        single detect
        """
        Gab = GAB(opt)
        Gab.LoadModel(opt.model_dir)

        path = "1.jpg"
        img = cv2.imread(path, 0)

        start = time.time()
        index, rects, scores = Gab.DetectFace(img)
        end = time.time()
        print ('use time:' + str(end - start))
        for i in range(len(index)):
            print ("{} {} {} {} {}".format(rects[index[i]][0], rects[index[i]][1], rects[index[i]][2], rects[index[i]][3], scores[index[i]]))
            if scores[index[i]] > 0:
                img = Gab.Draw(img, rects[index[i]])

        cv2.imwrite("2.jpg", img)

    def Live(self, opt):
        """
        Detect face from camera
        """
        cap = cv2.VideoCapture(0)

        Gab = GAB(opt)
        Gab.LoadModel(opt.model_dir)

        while(1):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            index, rects, scores = Gab.DetectFace(gray)

            for i in range(len(index)):
                if scores[index[i]] > 100:
                    frame = Gab.Draw(frame, rects[index[i]])

            cv2.imshow("live", frame)
            cv2.waitKey(0)
