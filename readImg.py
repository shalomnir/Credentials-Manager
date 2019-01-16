from PIL import Image
import numpy as np
import cv2 as cv
import simplejson as json
import sys
import glob
from functools import reduce

def GetImgHistogram(imgArr):
	
	img = (Image.fromarray(imgArr)).convert('L')
	#img = Image.open(img.filename).convert('L')  # convert image to 8-bit grayscale
	WIDTH, HEIGHT = img.size
	
	data = list(img.getdata()) # convert image data to a list of integers
	# convert that to 2D list (list of lists of integers)
	data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]
	
	# At this point the image's pixels are all in memory and can be accessed
	# individually using data[row][col].
	#print (data)
	histogram = [0] * 256
	# For example:
	for x in range(1, HEIGHT - 1):
		for y in range(1, WIDTH - 1):
			binList = ""
			curr = data[x][y]
			if( data[x-1][y-1] >= curr ):
				binList += '1'
			else:
				binList += '0'
			#	
			if(data[x-1][y] >= curr):
				binList += '1'
			else:
				binList += '0'
				
			#
			if(data[x-1][y+1] >= curr):
				binList += '1'
			else:
				binList += '0'
				
			#	
			if(data[x][y+1] >= curr):
				binList += '1'
			else:
				binList += '0'
				
			#	
			if(data[x+1][y+1] >= curr):
				binList += '1'
			else:
				binList += '0'
				
			#	
			if(data[x+1][y] >= curr):
				binList += '0'
			else:
				binList += '1'
				
			#	
			if(data[x+1][y-1] >= curr):
				binList += '1'
			else:
				binList += '0'
				
			#	
			if(data[x][y-1] >= curr):
				binList += '1'
			else:
				binList += '0'
			
			histogram[int(binList, 2)]+=1;
				
	return(histogram)
	
def ChiSquareCompering(H1,H2):
	d = 0
	for i in range(0,len(H1)):
		if(H1[i] != 0):
			d += np.square(H1[i]-H2[i])/H1[i]
	return d

	
def FaceCompare(img):
	d = []
	avg_d = -1
	with open('nirHistograms.json') as f:
		data = json.load(f)
	for hist in data['histogram']:
		d.append(ChiSquareCompering(GetImgHistogram(img),hist))
	avg_d = reduce(lambda x, y: x + y, d, 0) / len(d)	
	if(avg_d < 2000 and avg_d != -1):
		print ("AAAAAAAAAAAAAAAAAAA",avg_d)
		exit()
	print("NNNNNNNNNNNNNNNNN", avg_d)
	
def facechop():  
	cascPath = "haarcascade_frontalface_default.xml"
	faceCascade = cv.CascadeClassifier(cascPath)
	
	video_capture = cv.VideoCapture(0)
	
	while True:
		# Capture frame-by-frame
		ret, frame = video_capture.read()
		
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		
		faces = faceCascade.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(30, 30),
			flags=cv.CASCADE_SCALE_IMAGE
		)
		
		# Draw a rectangle around the faces
		for (x, y, w, h) in faces:
			cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
			crop_img = frame[y:y+h, x:x+w]
			cv.imshow('Video',frame)
			FaceCompare(crop_img)
				#cv.imshow('Video',crop_img)
				
			
		# Display the resulting frame
			#cv.imshow('Video',crop_img)
		
		if cv.waitKey(1) & 0xFF == ord('q'):
			break
		
		
	# When everything is done, release the capture
	video_capture.release()
	cv.destroyAllWindows()
	

	
print (facechop())
	
#print (cv.compareHist(np.array(GetImgHistogram('3.png')),np.array(GetImgHistogram('2.png')),  cv.HISTCMP_CHISQR))