import numpy as np
import cv2
from tkinter.filedialog import askopenfilename

prewitt_x = np.array ( [ [ -1  , 0 ,  1 ] ,
                         [ -1  , 0 ,  1 ] ,
                         [ -1  , 0 ,  1 ] ] )
prewitt_y = np.array ( [ [ -1 , -1 , -1 ] ,
                         [  0 ,  0 ,  0 ] ,
                         [  1 ,  1 ,  1 ] ] )

def prewitt_edge_detection ( input , prewitt_matrix ) :
    '''
            Algoritmus: Prewitt edge detection

        At each point in the image, the result of the Prewitt operator is either
        the corresponding gradient vector or the norm of this vector.
        In simple terms, the operator calculates the gradient of the image intensity
        at each point, giving the direction of the largest possible increase from
        light to dark and the rate of change in that direction.
    '''
    result = np.zeros( input.shape )
    for y in range( 0, input.shape[ 0 ] - 2 ) :
        for x in range( 0, input.shape[ 1 ] - 2 ) :
            result[ y ] [ x ] = np.sum( prewitt_matrix * input [ y : y + 3, x : x + 3 ] )  # 2-dimensional convolution operation
    return result

def non_maxima_suppress ( img , angle ) :
    '''
            Algoritmus: Non-maxima suppression

    1   From each position (x, y), step in the two directions
        perpendicular to edge orientation Θ(x, y)
    2   Denote inital pixel (x, y) by C, the two neighbouring pixels
        in perpendicular directions by A and B
    3   If M(A) > M(C) or M(B) > M(C), discard pixel (x, y) by
        setting M(x, y) = 0
    '''
    nms = np.zeros ( img.shape )
    for y in range ( 1 , img.shape [ 0 ] - 1 ) :
        for x in range ( 1 , img.shape [ 1 ] - 1 ) : # From each position (x, y), step in the two directions perpendicular to edge orientation Θ(x, y)
            if (angle [ y , x ] >= 0 and angle [ y , x ] < 22.5) :
                neighbour_value = max ( img [ y , x - 1 ] , img [ y , x + 1 ] )
            elif (angle [ y , x ] >= 22.5 and angle [y , x] < 67.5) :
                neighbour_value = max ( img [ y - 1 , x - 1 ] , img [ y + 1 , x + 1 ] )
            elif (angle [ y , x ] >= 67.5 and angle [ y , x ] < 112.5) :
                neighbour_value = max ( img [ y - 1 , x ] , img [ y + 1 , x ] )
            elif (angle [ y , x ] >= 112.5 and angle [ y , x ] < 157.5) :
                neighbour_value = max ( img [ y - 1 , x + 1 ] , img [ y + 1 , x - 1 ] )
            else :
                neighbour_value = max ( img [ y , x - 1 ] , img [ y , x + 1 ] )

            if img [ y , x ] < neighbour_value :
                nms [ y , x ] = 0  # If M(A) > M(C) or M(B) > M(C), discard pixel (x, y) by setting M(x, y) = 0
            else :
                nms [ y , x ] = img [ y , x ]
    return nms

if __name__ == "__main__":
    filename =  askopenfilename ( initialdir = "" ,
                                  title = "Select an image file" ,
                                  filetypes = [ ("Image Files" , "*.png") ] ) # input the image file
    img = cv2.imread( filename )
    cv2.imshow( 'Input', img )
    cv2.waitKey( 0 )
    gray_img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY ) # in case the input is not gray
    edge_x = prewitt_edge_detection( gray_img , prewitt_x )
    edge_y = prewitt_edge_detection( gray_img , prewitt_y )
    out = np.sqrt( edge_x ** 2 + edge_y ** 2 )
    out = (out / np.max ( out )) * 255 # mapping values from [0, 255]
    cv2.imshow( 'Prewitt edge-detection', out.astype( np.uint8 ) )
    cv2.waitKey( 0 )
    directions = np.rad2deg( np.arctan2 ( edge_y , edge_x ) ) # calculate the gradients' directions
    directions[ directions < 0 ] += 180
    nms = non_maxima_suppress ( out , directions )
    cv2.imshow( 'Non-maxima suppress on edges', nms.astype( np.uint8 ) )
    cv2.waitKey( 0 )