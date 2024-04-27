import numpy as np

matrix_1d = np.array([1,2,3])
matrix_2d = np.array([[1,2,3,4],[4,5,6,8],[7,8,9,6]])
matrix_3d = np.array([
        [
            [1,2,3],
            [4,5,6],
            [7,8,9]
         ],
         [             
                [1,2,3],
                [4,5,6],
                [7,8,9]       
         ]
    ])

# print(matrix_2d.shape)
# print(matrix_3d.shape)

# maxtrix_1d_padding = np.pad(matrix_1d, (2,4),'constant')
maxtrix_1d_padding = np.pad(matrix_1d, (2,4),'constant', constant_values=(3,1))
# print(maxtrix_1d_padding)
# print(maxtrix_1d_padding.shape)

# maxtrix_2d_padding = np.pad(matrix_2d, (2,4),'constant')
# maxtrix_2d_padding = np.pad(matrix_2d, (1,1),'constant')
maxtrix_2d_padding = np.pad(matrix_2d, ((1,2),(3,4)),'constant',
                            constant_values=((11,22),(33,44)))
# print(maxtrix_2d_padding)
# print(maxtrix_2d_padding.shape)
 
maxtrix_3d_padding = np.pad(matrix_3d, ((1,2),(2,3),(1,4)),'constant',
                            constant_values=((0,0),(22,33),(11,44)))
print(maxtrix_3d_padding)
print(maxtrix_3d_padding.shape)

