import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("originalimg", help="input the original image file name",
                    type=str)
# parser.add_argument("groundtruth", help="input the ground truth image file name",
#                     type=str)
args = parser.parse_args()

if args.originalimg=='image1prime.tif':
    ground=cv2.imread('GT_cup_1.png',0).astype(np.float32)/255
elif args.originalimg=='image2prime.tif':
    ground=cv2.imread('GT_cup_2.png',0).astype(np.float32)/255
elif args.originalimg=='image3prime.tif':
    ground=cv2.imread('GT_cup_3.png',0).astype(np.float32)/255
elif args.originalimg=='image4prime.tif':
    ground=cv2.imread('GT_cup_4.png',0).astype(np.float32)/255
elif args.originalimg=='image5prime.tif':
    ground=cv2.imread('GT_cup_5.png',0).astype(np.float32)/255
elif args.originalimg=='image6prime.tif':
    ground=cv2.imread('GT_cup_6.png',0).astype(np.float32)/255
elif args.originalimg=='image7prime.tif':
    ground=cv2.imread('GT_cup_7.png',0).astype(np.float32)/255
elif args.originalimg=='image8prime.tif':
    ground=cv2.imread('GT_cup_8.png',0).astype(np.float32)/255
elif args.originalimg=='image9prime.tif':
    ground=cv2.imread('GT_cup_9.png',0).astype(np.float32)/255
elif args.originalimg=='image10prime.tif':
    ground=cv2.imread('GT_cup_10.png',0).astype(np.float32)/255
elif args.originalimg=='image11prime.tif':
    ground=cv2.imread('GT_cup_11.png',0).astype(np.float32)/255
elif args.originalimg=='image12prime.tif':
    ground=cv2.imread('GT_cup_12.png',0).astype(np.float32)/255
elif args.originalimg=='image13prime.tif':
    ground=cv2.imread('GT_cup_13.png',0).astype(np.float32)/255
elif args.originalimg=='image14prime.tif':
    ground=cv2.imread('GT_cup_14.png',0).astype(np.float32)/255
elif args.originalimg=='image15prime.tif':
    ground=cv2.imread('GT_cup_15.png',0).astype(np.float32)/255
elif args.originalimg=='image16prime.tif':
    ground=cv2.imread('GT_cup_16.png',0).astype(np.float32)/255
elif args.originalimg=='image17prime.tif':
    ground=cv2.imread('GT_cup_17.png',0).astype(np.float32)/255
elif args.originalimg=='image18prime.tif':
    ground=cv2.imread('GT_cup_18.png',0).astype(np.float32)/255
elif args.originalimg=='image19prime.tif':
    ground=cv2.imread('GT_cup_19.png',0).astype(np.float32)/255
elif args.originalimg=='image20prime.tif':
    ground=cv2.imread('GT_cup_20.png',0).astype(np.float32)/255
elif args.originalimg=='image21prime.tif':
    ground=cv2.imread('GT_cup_21.png',0).astype(np.float32)/255
elif args.originalimg=='image22prime.tif':
    ground=cv2.imread('GT_cup_22.png',0).astype(np.float32)/255
elif args.originalimg=='image23prime.tif':
    ground=cv2.imread('GT_cup_23.png',0).astype(np.float32)/255
elif args.originalimg=='image24prime.tif':
    ground=cv2.imread('GT_cup_24.png',0).astype(np.float32)/255
elif args.originalimg=='image25prime.tif':
    ground=cv2.imread('GT_cup_25.png',0).astype(np.float32)/255
elif args.originalimg=='image26prime.tif':
    ground=cv2.imread('GT_cup_26.png',0).astype(np.float32)/255
elif args.originalimg=='image27prime.tif':
    ground=cv2.imread('GT_cup_27.png',0).astype(np.float32)/255
elif args.originalimg=='image28prime.tif':
    ground=cv2.imread('GT_cup_28.png',0).astype(np.float32)/255
elif args.originalimg=='image29prime.tif':
    ground=cv2.imread('GT_cup_29.png',0).astype(np.float32)/255
elif args.originalimg=='image30prime.tif':
    ground=cv2.imread('GT_cup_30.png',0).astype(np.float32)/255
# image=cv2.imread('image1prime.tif').astype(np.float32)/255
# ground=cv2.imread('GT_cup_3.png',0).astype(np.float32)/255
# ground=cv2.imread(args.groundtruth,0).astype(np.float32)/255
# image=cv2.imread('image3prime.tif')
image=cv2.imread(args.originalimg)
green=image[:,:,1]

# hist_green=plt.hist(green.ravel(),256,[0,256]); plt.show()

while True:
    try:
        #resize citra asli dan convert to grayscale-----------------
        r = 500.0 / green.shape[1]
        dim = (500, int(green.shape[0] * r))
        greengray = cv2.resize(green, dim, interpolation = cv2.INTER_AREA)
        # greyimage=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = greengray.copy()
        imagedisp = image.copy().astype(np.float32)/255
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(image)
        # cv2.circle(image, maxLoc, 20, (255, 0, 0), 2)
        # print(maxLoc, end="\r")
        roi=image[maxLoc[1]-32:maxLoc[1]+32, maxLoc[0]-32:maxLoc[0]+32].copy()
        # print(maxLoc[1]-5, end="\r")
        # image_en = cv2.equalizeHist(image)
        cv2.imshow('RGB', test2)
        # cv2.imshow('ROI', roi)

        #resize ground truth----------
        rg = 500.0 / ground.shape[1]
        dimg = (500, int(ground.shape[0] * rg))
        ground = cv2.resize(ground, dimg, interpolation = cv2.INTER_AREA)
        # cv2.imshow('ground',ground)
        
        # mean=np.mean(image)
        # std=np.std(image)
        # mean, std = cv2.meanStdDev(image)
        # print(mean, std, end="\r")
        # image = image-mean-std
        # cv2.imshow('img', image)
        test = roi.copy()
        element_type = cv2.MORPH_ELLIPSE
        element = cv2.getStructuringElement(element_type, (2*5+1, 2*5+1), (5, 5))
        blackhat = cv2.morphologyEx(test, cv2.MORPH_BLACKHAT, element)
        tophat = cv2.morphologyEx(test, cv2.MORPH_TOPHAT, element)
        roi = test + blackhat
        # cv2.imshow('blackhat', test)

        #histogram smoothing dengan gaussian blur untuk menonjolkan bagian dengan intensitas tinggi---------
        m=5
        # blur=cv2.GaussianBlur(image,(m,m),cv2.BORDER_DEFAULT)
        blurred=cv2.GaussianBlur(roi,(m,m),cv2.BORDER_DEFAULT)
        window=cv2.getGaussianKernel(m,0)
        # print(window)
        dummy=image.copy()
        dummy=cv2.rectangle(dummy, (0, 0), (int(dummy.shape[1]),int(dummy.shape[0])), (0, 0, 0), -1)
        dummy[maxLoc[1]-32:maxLoc[1]+32, maxLoc[0]-32:maxLoc[0]+32] = blurred
        dummy=dummy.copy().astype(np.float32)/255
        # blur=cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT).astype('uint8')
        (minVal2, maxVal2, minLoc2, maxLoc2) = cv2.minMaxLoc(blurred)
        mean, std = cv2.meanStdDev(blurred)
        # blurred=blurred-mean-std
        # cv2.imshow('blurred', blurred)
        # hist_blur=plt.hist(blur.ravel(),256,[0,256]); plt.show()
      


        #menghitung nilai threshold---------------
        sb=np.std(window)
        sg=std
        mg=mean
        T2 = (0.5*m) + (2*sb) + (2*sg) + mg
        if maxVal2<T2:
            diff=T2-maxVal2
            # print(diff,end="\r")
            blurred=blurred+(diff+5)
            # (minVal3, maxVal3, minLoc3, maxLoc3) = cv2.minMaxLoc(blurred)
            # print(maxVal3, T2, end="\r")
        else:
            pass
        # print(mean, std, maxVal2, T2, end="\r")

        #thresholding----------------
        # gaus = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,75,10)
        # (T,thres)=cv2.threshold(blur, T2, 255, cv2.THRESH_BINARY)
        (T,thres2)=cv2.threshold(blurred, T2, 255, cv2.THRESH_BINARY)
        # # cv2.imshow('image',gray)
        # thres2=thres2-mean-std
        # cv2.imshow('threshold',thres2)
        

        bg=image.copy()
        bg=cv2.rectangle(bg, (0, 0), (int(bg.shape[1]),int(bg.shape[0])), (0, 0, 0), -1)
        bg[maxLoc[1]-32:maxLoc[1]+32, maxLoc[0]-32:maxLoc[0]+32] = thres2
        # cv2.imshow('back',bg)

        #morphological operation----------
        # element_type = cv2.MORPH_ELLIPSE
        # element = cv2.getStructuringElement(element_type, (2*5+1, 2*5+1), (5, 5))
        # dilation = cv2.dilate(thres,element,iterations = 3)
        # element2 = cv2.getStructuringElement(element_type, (2*7+1, 2*7+1), (7, 7))
        # erosion = cv2.erode(dilation,element2,iterations = 2)
        # erosion2 = cv2.erode(erosion,element2,iterations = 1)
        # dilation2 = cv2.dilate(erosion2,element,iterations = 2)
        # erosion2 = cv2.erode(dilation2,element2,iterations = 1)

        element_type = cv2.MORPH_ELLIPSE
        element = cv2.getStructuringElement(element_type, (2*5+1, 2*5+1), (5, 5))
        bg = cv2.dilate(bg,element,iterations = 3)
        element2 = cv2.getStructuringElement(element_type, (2*7+1, 2*7+1), (7, 7))
        bg = cv2.erode(bg,element2,iterations = 2)
        bg = cv2.dilate(bg,element,iterations = 2)
        bg = cv2.erode(bg,element2,iterations = 1)

        # kernel1 = np.ones((1,1),np.uint8)
        # # opening = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel)
        # erosion = cv2.erode(thres,kernel1,iterations = 1)
        # kernel2 = np.ones((5,5),np.uint8)
        # dilation = cv2.dilate(erosion,kernel2,iterations = 1)
        # cv2.imshow('image2',bg)
        
        #menghitung korelasi------------
        segmented=bg.copy().astype(np.float32)/255
        cor = np.corrcoef(ground.reshape(-1), segmented.reshape(-1))[0][1]
        # cor = np.corrcoef(ground.reshape(-1), dilation.reshape(-1))[0][1]
        # cor = np.corrcoef(ground.reshape(-1), dilation.reshape(-1))[0][1]
        # print("correlation coefficient "+str(round(cor,5)),end="\r")
        
        # img = image.copy().astype(np.float32)/255
        # blurred = blur.copy().astype(np.float32)/255
        disp_ground=ground.copy()
        disp_segmented=segmented.copy()
        # disp_segmented=dilation.copy()
        # cv2.imshow('image',disp_ground)
        #display--------------
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(imagedisp,"Original Image (in grayscale)",(0,20), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(disp_ground,"Ground Truth",(0,20), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(dummy,"Preprocessed Image (ROI)",(0,20), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(disp_segmented,"Segmented Image",(0,20), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(disp_segmented,"correlation coefficient "+str(round(cor,5)),(0,40), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

        ori=np.concatenate((imagedisp, disp_ground), axis=1)
        # processed=np.concatenate((blur, dilation), axis=1)
        processed=np.concatenate((dummy, disp_segmented), axis=1)
        joined=np.concatenate((ori,processed), axis=0)
        cv2.imshow('Optic Cup Segmentation',joined)

        cv2.waitKey(1)
        # print(T2, sg, mg)
    except KeyboardInterrupt:
        break

cv2.destroyAllWindows()