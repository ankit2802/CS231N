import numpy
from PIL import Image


def main():
    img = numpy.array(Image.open('C:/Users/Gagandeep/Desktop/Gaga/studies/2nd sem/Data Mining/HWs/HW11/Q1.jpeg').getdata(),
                    numpy.uint8)
    #img=img.reshape(img.size[0], img.size[1], 3)
    #arr = PIL2array(img)
    print(img.shape)
    #img2 = array2PIL(arr, img.size)
    #img2.save('out.jpg')

if __name__ == '__main__':
    main()