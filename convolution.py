# https://www.youtube.com/watch?v=swguqT77ZLE&list=PL8ZSveYn9kVT0DcOXQKcnuhLlCZeG3a-k

import cv2
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

img = cv2.imread('person.jpeg')
# print(img.shape)
img = cv2.resize(img,(200,200))
# print(img.shape)

# Convert color to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
# print(img_gray.shape)

class Conv2d:
    def __init__(self, input, numOfKernel=8, kernel_size=3, padding=0, stride=1):
        # self.input = input
        self.input = np.pad(input, ((padding,padding),(padding,padding)),'constant')
        self.stride = stride
        # Random kernel value
        self.kernel = np.random.randn(numOfKernel, kernel_size, kernel_size)  
        print(self.kernel.shape)
        # Initialize the result matrix
        # There is a number of kernels, we need to update something
        self.result = np.zeros( 
                                (   
                                    int((self.input.shape[0] - kernel_size)/stride) + 1, 
                                    int((self.input.shape[1] - kernel_size)/stride) + 1,
                                    self.kernel.shape[0]
                                )
                            )

    # for row in range(0, height-kernel_size + 1):
    #     for col in range(0, width-kernel_size + 1):
    #         # roi: region of interesting
    #         result[row,col] = np.sum(input[row: row + kernel_size, col:col + kernel_size] * kernel)

    # Refactor code
    def getROI(self): 
        kernel_size = self.kernel.shape[1]
        for row in range(0, int((self.input.shape[0]-kernel_size)/self.stride) + 1):
            for col in range(0, int((self.input.shape[1]-kernel_size)/self.stride) + 1):
                roi = self.input[row * self.stride: row * self.stride + kernel_size, 
                                 col* self.stride: col* self.stride + kernel_size]
                # Cannot use return, it will end the function
                # return row, col, roi
                yield row, col, roi

    def operate(self):
        # Loop through the number of kernels
        for layer in range(self.kernel.shape[0]):
            for row, col, roi in self.getROI( ):
                # image shape format: (width, height, layer_size)
                self.result[row,col, layer] = np.sum(roi * self.kernel[layer, :, :])
        return self.result

class Relu:
    def __init__(self, input):
        self.input = input
        # width, height, shape
        self.result = np.zeros((self.input.shape[0], self.input.shape[1], self.input.shape[2]))

    def operate(self):
        # Loop through the number of kernels
        for layer in range(self.input.shape[2]):
            for row in range(self.input.shape[0]):
                for col in range(self.input.shape[1]):
                    # relu formula
                    self.result[row,col,layer] = 0 if self.input[row,col, layer]  < 0 else self.input[row,col, layer]
        return self.result

class LeakyRelu:
    def __init__(self, input):
        self.input = input
        # width, height, shape
        self.result = np.zeros((self.input.shape[0], self.input.shape[1], self.input.shape[2]))

    def operate(self):
        # Loop through the number of kernels
        for layer in range(self.input.shape[2]):
            for row in range(self.input.shape[0]):
                for col in range(self.input.shape[1]):
                    # relu formula
                    self.result[row,col,layer] = 0.1 * self.input[row,col, layer] if self.input[row,col, layer]  < 0 else self.input[row,col, layer]
        return self.result

# # conv2d = Conv2d(img_gray,3,2,1)
# # conv2d = Conv2d(img_gray,5,padding=4,stride=3)
# conv2d = Conv2d(img_gray,3)
# img_gray_conv2d = conv2d.operate()
# plt.imshow(img_gray_conv2d, cmap='gray')
# # print(img_gray_conv2d.shape)
# plt.show()

# # Add Relu
# conv2d_relu = Relu(img_gray_conv2d)
# img_gray_conv2d_relu = conv2d_relu.operate()
# plt.imshow(img_gray_conv2d_relu, cmap='gray')
# # print(img_gray_conv2d.shape)
# plt.show()

# # Create 9 pictures
# for i in range(9):
#     conv2d = Conv2d(img_gray, 3, padding=2, stride=1)
#     img_gray_conv2d = conv2d.operate()
#     conv2d_relu = Relu(img_gray_conv2d)
#     img_gray_conv2d_relu = conv2d_relu.operate()

#     plt.subplot(3, 3 , i+ 1)
#     plt.imshow(img_gray_conv2d_relu, cmap='gray')
    
# plt.show()

# conv2d = Conv2d(img_gray, numOfKernel=8, kernel_size=3, padding=0, stride=1)
# conv2d = Conv2d(img_gray, numOfKernel=16, kernel_size=3, padding=0, stride=1)
# img_gray_conv2d = conv2d.operate()


# for i in range(16):
#     plt.subplot(4, 4, i + 1)
#     plt.imshow(img_gray_conv2d[:, :, i], cmap='gray')
#     plt.axis('off')
# plt.savefig('img_gray_conv2d.jpg') 
# plt.show()

conv2d = Conv2d(img_gray, numOfKernel=16, kernel_size=3, padding=0, stride=1)
img_gray_conv2d = conv2d.operate()
img_gray_conv2d_relu = Relu(img_gray_conv2d).operate()
img_gray_conv2d_leaky_relu = LeakyRelu(img_gray_conv2d).operate()

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(img_gray_conv2d_leaky_relu[:, :, i], cmap='gray')
    plt.axis('off')
plt.savefig('img_gray_conv2d_leaky_relu.jpg')
plt.show()
