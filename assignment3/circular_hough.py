import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import math
from collections import defaultdict

def circular_detection ( image , diameterRange ) :
    '''
    :param image:  image containing circular objects
    :param diameterRange: range of diameters
    :return: 1. accumulator image; 2. input image with objects detected in given range of diameters.
    '''
    radiusRange = [ int ( elem / 2 ) for elem in diameterRange ]
    rangeMin = radiusRange [ 0 ]
    rangeMax =  radiusRange [ 1 ]
    row , column = image.shape

    accumulatorMatrix = np.zeros ( [ 2 * row , 2 * column , rangeMax + 1 ] )
    for r in radiusRange :
        for x in range ( row ) :
            for y in range ( column ) :
                if image [ x ] [ y ] != 0 :
                    for theta in range ( 0 , 360 ) :
                        b = y - r * np.sin ( theta * math.pi / 180 )
                        a = x - r * np.cos ( theta * math.pi / 180 )
                        accumulatorMatrix [ int ( a ) , int ( b ) , int ( r ) ] += 1  # voting

    accumulatorMatrix_ = np.sum ( accumulatorMatrix , axis = 2 )
    accumulatorMatrix_ = ( ( accumulatorMatrix_ - accumulatorMatrix_.min() ) *
                           ( 1.0 / (accumulatorMatrix_.max() - accumulatorMatrix_.min() ) * 255 ) ) # normalization

    accumulatorImage = accumulatorMatrix_ [ : row + 1 , : column + 1 ]
    max_value = np.max ( accumulatorMatrix )
    indices = np.argwhere ( accumulatorMatrix > ( max_value * 0.7 ) )

    return indices, accumulatorImage

if __name__ == "__main__":
    filename =  askopenfilename ( initialdir = "" ,
                                  title = "Select an image file" ,
                                  filetypes = [ ("Image Files" , "*.png") ] ) # input the image file
    img = cv2.imread( filename )
    edgeDetected = cv2.Canny ( img , 50 , 150 )

    indices, accumulatorImage = circular_detection( edgeDetected , [ 15 , 25 ] ) # circles 15-25

    for x , y , r in indices :
        cv2.circle ( img , ( y , x ) , r , (0 , 0 , 255) , thickness = 1 , lineType = 8 , shift = 0 )

    cv2.imshow ( 'Image' , img )
    cv2.imshow ( 'Accumulator Image' , accumulatorImage.astype ( np.uint8 ) )
    cv2.waitKey ( 0 )