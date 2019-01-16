
import glob
import cv2 as cv
from PIL import Image
import simplejson as json
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



def facechop():  
	cascPath = "haarcascade_frontalface_default.xml"
	faceCascade = cv.CascadeClassifier(cascPath)
	f = open("nirHistograms.json", "a")
	data = {
				"histogram": []
			}	
	video_capture = cv.VideoCapture(0)
	count = 0;
	while(count < 10):
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
			data["histogram"].append(GetImgHistogram(crop_img))
			count+=1
				#cv.imshow('Video',crop_img)
				
			
		# Display the resulting frame
			#cv.imshow('Video',crop_img)
		
		if cv.waitKey(1) & 0xFF == ord('q'):
			break
		
	
	y = json.dumps(data)
	f.write(y)
	# When everything is done, release the capture
	video_capture.release()
	cv.destroyAllWindows()
	

def loadJson():
	with open('nirHistograms.json') as f:
			data = json.load(f)
	
	for hist in data[histogram]:
		ChiSquareCompering(GetImgHistogram(img),GetImgHistogram(hist))
	print(data)

def CreateJson():
	f = open("nirHistograms.json", "a")
	data = {
				"histogram": []
			}	
	for filename in glob.glob(r'C:\Users\nirsh\Desktop\photos\nirImages\*.png'):		
			data["histogram"].append(GetImgHistogram(cv.imread(filename, 0)))
			y = json.dumps(data)
			f.write(y)
		
facechop()