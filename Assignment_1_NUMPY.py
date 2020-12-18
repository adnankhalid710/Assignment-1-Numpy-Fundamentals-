# Assignment#1(Numpy Fundamentals)
# https://github.com/adnankhalid710/AI-Q2-learning-resources/blob/master/NumpyAssignments/Assignment%231(Numpy%20Fundamentals).ipynb
# 1-Import the numpy package under the name np
import numpy as np
'''
# -------------------------------------------------------------------
# Difficulty Level Medium
# 2-Create a null vector of size 10
a1 = np.zeros(10)
# print(a1)
# 3-Create a vector with values ranging from 10 to 49
a2 = np.arange(10, 49)
# print(a2)
# 4-Find the shape of previous array in question 3
print(a2.shape)
# 5-Print the type of the previous array in question 3
print(type(a2))
# 6-Print the numpy version and the configuration
print(np.__version__)
# 7-Print the dimension of the array in question 3
print(a2.shape)
# 8-Create a boolean array with all the True values
a3 = np.ones(2)
# print(a3)

# 9-Create a two dimensional arra
arr2d = np.array([[2,3],[8,6]])
# print(arr2d)
# print(arr2d.ndim)

# 10-Create a three dimensional array
arr3d = np.array([[[1,2,2],[3,3,4],[32,33,34]]])
print(arr3d)
# arr3da = np.arange(27).reshape(3,3,3)
# print(arr3d)
'''
# -------------------------------------------------------------------
# Difficulty Level Easy
'''
# 11-Reverse a vector (first element becomes last)
v1 = np.arange(0,10)
print(f"The real vector is as follow{v1}")
r_v1 = np.flipud(v1)
print(f"The reverse vector is as follow{r_v1}")

# 12-Create a null vector of size 10 but the fifth value which is 1
null_vec = np.zeros(10)
null_vec[5] = 1
print(null_vec)

# 13-Create a 3x3 identity matrix
id_mat = np.identity(2)
print(id_mat)

# 14- arr = np.array([1, 2, 3, 4, 5])
# Convert the data type of the given array from int to float
arr = np.array([1, 2, 3, 4, 5], np.float32)
print(arr.dtype)

# 15-Multiply arr1 with arr2
arr1 = np.array([[1., 2., 3.],[4., 5., 6.]])
arr2 = np.array([[0., 4., 1.],[7., 2., 12.]])
print(arr1*arr2)

# 16-Make an array by comparing both the arrays provided above
print(arr1 == arr2)

# 17-Extract all odd numbers from arr with values(0-9)
b1 = np.arange(0,9)
print(b1)
odd_v_b1 = b1[b1 % 2 == 1]
print(f"Odd values from above vector is as follow \n {odd_v_b1}")

# 18-Replace all odd numbers to -1 from previous array
b1[b1 % 2 == 1] = -1
print(b1)

# 19-Replace the values of indexes 5,6,7 and 8 to 12
c1 = np.arange(0,9)
print(f"The rel array is as follow \n {c1}")
c1[5:] = 12
print(f"Replace the values of indexes 5,6,7 and 8 to 12 is as follow \n {c1}")

# 20-Create a 2d array with 1 on the border and 0 inside
'''
'''
c_2d = np.zeros(16).reshape(4,4)
c_2d[0:4, 0:1] = 1  # first row    [row:row, Column:column]
c_2d[0:1, 0:4] = 1  # first column
c_2d[0:4, 3:4] = 1  # last column
c_2d[3:4, 0:4] = 1  # last row
'''
'''
c_2d = np.zeros(25).reshape(5, 5)
c_2d[0:, 0:1] = 1  # first row
c_2d[0:1, 0:] = 1  # first column

c_2d[0:, -1:] = 1  # last column
c_2d[-1:, 0:] = 1  # last row
print(c_2d)
'''
# -------------------------------------------------------------------
# Difficulty Level Medium
# 21-Replace the value 5 to 12
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[1:2, 1:2]= 12
print(arr2d)

# 22-Convert all the values of 1st array to 64
print("-------------------------------\n")
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0]= 64
print(arr3d)

# 23-Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it
ar2d = np.arange(9).reshape(3,3)
print(ar2d[0])

# 24-Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it
print(ar2d[1])

# 25-Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

print(ar2d[0:2, 2:])

# 26-Create a 10x10 array with random values and find the minimum and maximum values
from numpy import random
abc = random.randint(100, size = (10, 10))
print("array 10 by 10")
print(abc)
mini_value = np.amin(abc)
maxi_value = np.amax(abc)
print(f"Minimum value in above created array is {mini_value}")
print(f"Maximum value in above created array is {maxi_value}")

# 27-Find the common items between a and b
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(np.intersect1d(a, b))

# 28-Find the positions where elements of a and b match
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(f"Positions of common values in above two arrays {np.arange(len(a))[a==b]}")

# 29-Find all the values from array data where the values from array names are not equal to Will
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
print(data[names != 'Will'])

# 30-Find all the values from array data where the values from array names are not equal to Will and Joe
mask = (names != 'Joe') & (names != 'Will')
print(data[mask])


# -------------------------------------------------------------------
# Difficulty Level Hard
# 31-Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15

b_arr = np.random.uniform(5,10, size=(5,3))
print(b_arr)

# 32-Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16
c_arr = np.random.uniform(2, 8, size=(2, 2, 4))
print(c_arr)

# 33-Swap axes of the array you created in Question 32


# 34-Create an array of size 10, and find the square root of every element in the array,
# if the values less than 0.5, replace them with 0
bca = np.array([0.2, 49, 9, 4, 144, 36, 64, 121, 169, 100])
print(bca)
acb = np.sqrt(bca)
print(np.where(acb < 0.5, 1, acb))

# 35-Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# 36-Find the unique names and sort them out!

# 37-From array a remove all items present in array b
a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 6, 7, 8, 9])

