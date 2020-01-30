import os
import numpy as np
import math
import struct
import cv2

class GAB:
    def __init__(self, opt):
        """
        Init Feature map
        Not only generate feature map and also generate feature-coordinate map
        """
        # defined detect window size
        self.pWinSize = [24, 29, 35, 41, 50,
                         60, 72, 86, 103, 124,
                         149, 178, 214, 257, 308,
                         370, 444, 532, 639, 767,
                         920, 1104, 1325, 1590, 1908,
                         2290, 2747, 3297, 3956]
        # vector contain point-feature map
        self.lpoints = []
        self.rpoints = []

        # save the points of feature id
        self.points1 = [[] for i in range(29)]
        self.points2 = [[] for i in range(29)]

        self.opt = opt

        # A feature map used for speed up calculate feature
        self.ppNpdTable = np.zeros((256, 256))

        for i in range(256):
            for j in range(256):
                fea = 0.5
                if i > 0 or j > 0:
                    fea = i / (i + j)
                fea = math.floor(256 * fea)
                fea = min(fea, 255)

                self.ppNpdTable[i,j] = int(fea)

        numPixels = self.opt.objSize * self.opt.objSize

        for i in range(numPixels):
            for j in range(i+1,numPixels):
                self.lpoints.append(i)
                self.rpoints.append(j)


    def GetPoints(self, feaId):
        """ 
        Get the coordinates by feature id
        the feature number is calculate by (objSize*objSize)*(objSize*objSize-1)/2
        so if you have a feature id, use this function to get the coordinates
        got the coordinates you can calculate the feature value in image.
        here use two maps which store feature-coordinates
        
        param feaid  Feature Id
        return 
        x1  coordinate of point A.x
        x2  coordinate of point A.y
        y1  coordinate of point B.x
        y2  coordinate of point B.y
        """
        lpoint = self.lpoints[feaId]
        rpoint = self.rpoints[feaId]
        x1 = int(lpoint / self.opt.objSize)
        y1 = int(lpoint % self.opt.objSize)
        x2 = int(rpoint / self.opt.objSize)
        y2 = int(rpoint % self.opt.objSize)
        return [x1, y1, x2, y2]

    
    def LoadModel(self, path):
        """
        Load Model Which Trained by []
        
        param path  the model path
        """
        if not os.path.exists(path):
            return False

        f = open(path, 'rb')
        # model template size
        DetectSize = struct.unpack('i', f.read(4))[0]
        # indicate how many stages the dector have
        self.stages = struct.unpack('i', f.read(4))[0]
        numBranchNodes = struct.unpack('i', f.read(4))[0]
        
        # vectors contain the model
        self.treeIndex = []
        self.thresholds = []
        self.feaIds = []
        self.leftChilds = []
        self.rightChilds = []
        self.cutpoints = []
        self.fits = []

        for i in range(self.stages):
            self.treeIndex.append(struct.unpack('i', f.read(4))[0])
        for i in range(self.stages):
            self.thresholds.append(struct.unpack('f', f.read(4))[0])
        for i in range(numBranchNodes):
            self.feaIds.append(struct.unpack('i', f.read(4))[0])
        for i in range(numBranchNodes):
            self.leftChilds.append(struct.unpack('i', f.read(4))[0])
        for i in range(numBranchNodes):
            self.rightChilds.append(struct.unpack('i', f.read(4))[0])
        for i in range(2 * numBranchNodes):
            self.cutpoints.append(ord(struct.unpack('c', f.read(1))[0]))
        for i in range(numBranchNodes):
            for j in range(29):
                x1, y1, x2, y2 = self.GetPoints(self.feaIds[i])
                factor = self.pWinSize[j] / DetectSize;
                p1x = int(x1 * factor)
                p1y = int(y1 * factor)
                p2x = int(x2 * factor)
                p2y = int(y2 * factor)
                self.points1[j].append(p1y * self.pWinSize[j] + p1x)
                self.points2[j].append(p2y * self.pWinSize[j] + p2x)
        
        numLeafNodes = numBranchNodes + self.stages
        for i in range(numLeafNodes):
            fit = struct.unpack('f', f.read(4))[0]
            self.fits.append(fit)

        f.close()

        return True
    
    
    def Draw(self, img, rects):
        """
        Draw rect in a image
        
        param img  the image need to be draw
        param rects  the box
        """
        img = cv2.rectangle(img, rects, (0, 0, 255), 2)
        return img


    def DetectFace(self, img):
        """
        wraper for Detect faces from a image
        Sliding and resize window to scrach all the regions
        return a vector which save the index of face regions
        
        param img  The image need to be detected
        """
        minFace = 20
        maxFace = 1000

        #The vector that contain the location of faces
        rects = []
        #the vector thar contain the faces score
        scores = []

        height, width = img.shape
        sizePatch = min(width, height)
        thresh = 0

        img = img.transpose(1, 0)
        
        I = img.flatten()

        for i in range(29):
            if sizePatch >= self.pWinSize[i]:
          	  thresh = i
            else:
          	  break

        # the total no. of scales that will be searched
        thresh += 1 

        minFace = max(minFace, self.opt.objSize)
        maxFace = min(maxFace, sizePatch)

        picked = []
        if sizePatch < minFace:
          return picked

        # process each scale
        for k in range(thresh):
            if self.pWinSize[k] < minFace:
                continue
            elif self.pWinSize[k] > maxFace:
                break

            # determine the step of the sliding subwindow
            winStep = math.floor(self.pWinSize[k] * 0.1)
            if self.pWinSize[k] > 40:
                winStep = math.floor(self.pWinSize[k] * 0.05)

            # calculate the offset values of each pixel in a subwindow
            # pre-determined offset of pixels in a subwindow
            offset = [0 for i in range(self.pWinSize[k] * self.pWinSize[k])]
            pp1 = 0 
            pp2 = 0 
            gap = height - self.pWinSize[k]

            for j in range(self.pWinSize[k]):
                for i in range(self.pWinSize[k]):
                    offset[pp1] = pp2
                    pp2 += 1
                    pp1 += 1
                pp2 += gap

            colMax = width - self.pWinSize[k] + 1
            rowMax = height - self.pWinSize[k] + 1

            # process each subwindow
            for c in range(0, colMax, winStep):
                # slide in column
                pPixel = c * height

                for r in range(0, rowMax, winStep):
                    # slide in row
                    _score = 0

                    # test each tree classifier
                    for s in range(self.stages):
                        node = self.treeIndex[s]
                        #print (k, c, r, s, node)
                        #7 12 4 4

                        # test the current tree classifier
                        # branch node
                        while node > -1:
                            p1 = I[pPixel + offset[self.points1[k][node]]]
                            p2 = I[pPixel + offset[self.points2[k][node]]]
                            fea = self.ppNpdTable[p1, p2]

                            if fea < self.cutpoints[2 * node] or fea > self.cutpoints[2 * node + 1]:
                                node = self.leftChilds[node]
                            else: 
                                node = self.rightChilds[node]

                        node = - node - 1
                        _score = _score + self.fits[node]

                        # negative samples
                        if _score < self.thresholds[s]:
                            break
                    # a face detected
                    if s == self.stages - 1:
                        roi = [c, r, self.pWinSize[k], self.pWinSize[k]]
                        # modify the record by a single thread
                        rects.append(roi)
                        scores.append(_score)

                    pPixel += winStep

        picked, rects, scores, Srect = self.Nms(rects, scores, 0.5, img)

        # you should set the parameter by yourself
        for i in range(len(picked)):
            idx = picked[i]
            delta = math.floor(Srect[idx] * self.opt.enDelta)
            y0 = max(rects[idx][1] - math.floor(3.0 * delta), 0)
            y1 = min(rects[idx][1] + Srect[idx], height)
            x0 = max(rects[idx][0] + math.floor(0.25 * delta), 0)
            x1 = min(rects[idx][0] + Srect[idx] - math.floor(0.25 * delta), width)

            rects[idx][1] = y0
            rects[idx][0] = x0
            rects[idx][2] = x1 - x0 + 1
            rects[idx][3] = y1 - y0 + 1

        return picked, rects, scores


    def Nms(self, rects, scores, overlap, img):
        """
        nms Non-maximum suppression
        the Nms algorithm result concerned score of areas
         
        param rects     area of faces
        param scores    score of faces
        param overlap   overlap threshold
        param img  get size of origin img
        return          picked index
        """
        numCandidates = len(rects)
        predicate = np.eye(numCandidates, numCandidates)
        for i in range(numCandidates):
            for j in range(i+1, numCandidates):
                h = min(rects[i][1] + rects[i][3], rects[j][1] + rects[j][3]) - max(rects[i][1], rects[j][1])
                w = min(rects[i][0] + rects[i][2], rects[j][0] + rects[j][2]) - max(rects[i][0], rects[j][0])
                s = max(h, 0) * max(w, 0)

                if s / (rects[i][2] * rects[i][3] + rects[j][2] * rects[j][3] - s) >= overlap:
                    predicate[i, j] = 1
                    predicate[j, i] = 1

        numLabels, label = self.Partation(predicate)

        Rects = []
        Srect = []
        neighbors = []
        Score = []

        for i in range(numLabels):
            index = []
            for j in range(numCandidates):
                if label[j] == i:
                    index.append(j)

            weight = self.Logistic(scores, index)

            sumScore = 0
            for j in range(len(weight)):
                sumScore += weight[j]
            Score.append(sumScore)
            neighbors.append(len(index))

            if sumScore == 0:
                for j in range(len(weight)):
                    weight[j] = 1 / sumScore
            else:
                for j in range(len(weight)):
                    weight[j] = weight[j] / sumScore

            size = 0
            col = 0
            row = 0
            for j in range(len(index)):
                size += rects[index[j]][2] * weight[j]

            Srect.append(math.floor(size))
            for j in range(len(index)):
                col += (rects[index[j]][0] + rects[index[j]][2] / 2) * weight[j]
                row += (rects[index[j]][1] + rects[index[j]][2] / 2) * weight[j]

            x = math.floor(col - size / 2)
            y = math.floor(row - size / 2)
            roi = [x, y, Srect[i], Srect[i]]
            Rects.append(roi)

        predicate = np.zeros((numLabels, numLabels))

        for i in range(numLabels):
            for j in range(numLabels):
                h = min(Rects[i][1] + Rects[i][3], Rects[j][1] + Rects[j][3]) - max(Rects[i][1], Rects[j][1])
                w = min(Rects[i][0] + Rects[i][2], Rects[j][0] + Rects[j][2]) - max(Rects[i][0], Rects[j][0])
                s = max(h, 0) * max(w, 0)

                if s / (Rects[i][2] * Rects[i][3]) >= overlap or \
                    s / (Rects[j][2] * Rects[j][3]) >= overlap:
                    predicate[i, j] = 1
                    predicate[j, i] = 1

        flag = [1 for i in range(numLabels)]

        for i in range(numLabels):
            index = []
            for j in range(numLabels):
                if predicate[j,i] == 1:
                    index.append(j)
            if not index:
                continue

            s = 0
            for j in range(len(index)):
                if Score[index[j]] > s:
                    s = Score[index[j]]
            if s > Score[i]:
                flag[i] = 0

        picked = []
        for i in range(numLabels):
            if flag[i]:
                picked.append(i)

        height, width = img.shape

        for i in range(len(picked)):
            idx = picked[i]
            if Rects[idx][0] < 0:
              Rects[idx][0] = 0

            if Rects[idx][1] < 0:
              Rects[idx][1] = 0

            if Rects[idx][1] + Rects[idx][3]> height:
              Rects[idx][3] = height - Rects[idx][1]

            if Rects[idx][0] + Rects[idx][2] > width:
              Rects[idx][2] = width - Rects[idx][0]

        return picked, Rects, Score, Srect

    
    def Partation(self, predicate):
        """
        function for Partation areas
        From Predicate mat get a paration result
        
        param predicate  The matrix marked cross areas
        return nGroups: number of classfication 
               label: The vector marked classification label
        """
        _, N = predicate.shape
        parent = [i for i in range(N)]
        rank = [0 for i in range(N)]

        for i in range(N):
            for j in range(N):
                if predicate[i,j] == 0:
                    continue
                root_i = self.Find(parent, i)
                root_j = self.Find(parent, j)

                if root_j != root_i:
                    if rank[root_j] < rank[root_i]:
                        parent[root_j] = root_i
                    elif rank[root_j] > rank[root_i]:
                        parent[root_i] = root_j
                    else:
                        parent[root_j] = root_i
                        rank[root_i] = rank[root_i] + 1

        nGroups = 0
        label = []
        for i in range(N):
            if parent[i]==i:
                label.append(nGroups)
                nGroups += 1
            else:
                label.append(-1)

        for i in range(N):
            if parent[i] == i:
                continue
            root_i = self.Find(parent, i)
            label[i] = label[root_i]

        return nGroups, label

    
    def Find(self, parent, x):
        """
        Find classfication area parent
        
        param parent  parent vector
        param x  current node
        """
        root = parent[x]
        if root != x:
          root = self.Find(parent, root)
        return root


    def Logistic(self, scores, index):
        """
        Compute score
        y = log(1+exp(x));
        
        param scores  score vector
        param index  score index
        """
        Y = []
        for i in range(len(index)):
            tmp_Y = math.log(1 + math.exp(scores[index[i]]))
            if math.isinf(tmp_Y):
                tmp_Y = scores[index[i]]
            Y.append(tmp_Y)
        return Y
