# https://www.youtube.com/watch?v=swguqT77ZLE&list=PL8ZSveYn9kVT0DcOXQKcnuhLlCZeG3a-k

import cv2
import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(42)

img = cv2.imread('person.jpeg')
# print(img.shape)
img = cv2.resize(img,(200,200))
# print(img.shape)

# Convert color to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
print(img_gray.shape)

class Conv2d:
    def __init__(self, input, kernel_size, padding=0, stride=1):
        # self.input = input
        self.input = np.pad(input, ((padding,padding),(padding,padding)),'constant')
        self.stride = stride
        # Random kernel value
        self.kernel = np.random.randn(kernel_size, kernel_size)  
        print(self.kernel)
        # Initialize the result matrix
        # print(int((input.shape[0] - kernel_size)/stride))
        # print(int((input.shape[1] - kernel_size)/stride))
        self.result = np.zeros(
                                (   
                                    int((self.input.shape[0] - kernel_size)/stride) + 1, 
                                    int((self.input.shape[1] - kernel_size)/stride) + 1
                                )
                            )

    # for row in range(0, height-kernel_size + 1):
    #     for col in range(0, width-kernel_size + 1):
    #         # roi: region of interesting
    #         result[row,col] = np.sum(input[row: row + kernel_size, col:col + kernel_size] * kernel)

    # Refactor code
    def getROI(self):
        kernel_row = self.kernel.shape[0]
        kernel_col = self.kernel.shape[1]
        for row in range(0, int((self.input.shape[0]-kernel_row)/self.stride) + 1):
            for col in range(0, int((self.input.shape[1]-kernel_col)/self.stride) + 1):
                roi = self.input[row * self.stride: row * self.stride + kernel_row, 
                                 col* self.stride: col* self.stride + kernel_col]
                # Cannot use return, it will end the function
                # return row, col, roi
                yield row, col, roi

    def operate(self):
        for row, col, roi in self.getROI( ):
            self.result[row,col] = np.sum(roi * self.kernel)
        return self.result

    
# conv2d = Conv2d(img_gray,3,2,1)
conv2d = Conv2d(img_gray,5,padding=4,stride=3)
img_gray_conv2d = conv2d.operate()

plt.imshow(img_gray_conv2d, cmap='gray')
print(img_gray_conv2d.shape)
plt.show()