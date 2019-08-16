#!/usr/bin/env python
# coding: utf-8


import random as r
class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.matrix = []
        
        for i in range(self.rows):
            self.matrix.append([])
            for j in range(self.cols):
                self.matrix[i].append(0)
    
    def printMatrix(self):
        for x in range(len(self.matrix)):
            print(self.matrix[x])
    
    @staticmethod
    def multiply(a, b):
        # Matrix or Dot Multiplication
        if a.cols != b.rows:
            print("Columns of A must be match Rows of B")
            return None
        result = Matrix(a.rows, b.cols)
        for i in range(a.rows):
            for j in range(b.cols):
                for k in range(b.rows):
                    result.matrix[i][j] += a.matrix[i][k] * b.matrix[k][j]      
        return result
        
    def scale(self, n):
        if type(n) == Matrix:
            for row in range(n.rows):
                for col in range(n.cols):
                    self.matrix[row][col] *= n.matrix[row][col]
        # Scalar Multiplication
        elif type(n) == int or type(n) == float: 
            for row in range(self.rows):
                for col in range(self.cols):
                    self.matrix[row][col] *= n
        else:
            print("tHis is a scalar function")

            
    def add(self, n):
        # Element-wise Addition
        if type(n) == Matrix:
            for row in range(self.rows):
                for col in range(self.cols):
                    self.matrix[row][col] += n.matrix[row][col]
        # Scalar Addition
        elif type(n) == int or type(n) == float:
            for row in range(self.rows):
                for col in range(self.cols):
                    self.matrix[row][col] += n
    
    def randomize(self):
        for row in range(self.rows):
            for cols in range(self.cols):
                self.matrix[row][cols] = (r.random() * 2) - 1
    
    @staticmethod
    def transpose(a):
        result = Matrix(a.cols, a.rows)
        for row in range(a.rows):
            for col in range(a.cols):
                result.matrix[col][row] = a.matrix[row][col]
        return result
    
    def mapThis(self, func):
        for row in range(self.rows):
            for col in range(self.cols):
                val = self.matrix[row][col]
                self.matrix[row][col] = func(val)

    @staticmethod
    def mapThisStatic(matrix, func):
        result = Matrix(matrix.rows, matrix.cols)
        for row in range(matrix.rows):
            for col in range(matrix.cols):
                val = matrix.matrix[row][col]
                result.matrix[row][col] = func(val)
        return result
          
    @staticmethod
    def fromArray(arr):
        m = Matrix(len(arr), 1)
        for i in range(len(arr)):
            m.matrix[i][0] = arr[i]
        return m
    
    def toArray(self):
        result = []
        for row in range(self.rows):
            for col in range(self.cols):
                result.append(self.matrix[row][col])
        return result
    
    @staticmethod
    def subtract(a , b):
        # Element-wise Subtraction
        if a.cols != b.cols:
            print("Cols should match")
        result = Matrix(a.rows, a.cols)
        if type(a) == Matrix:
            for row in range(a.rows):
                for col in range(a.cols):
                    result.matrix[row][col] = a.matrix[row][col] - b.matrix[row][col]
        return result
    
if __name__ == "__main__":
    m1 = Matrix(3, 3)
    m1.randomize()
    m1.printMatrix()

    m2 = Matrix(3, 3)
    m2.randomize()
    m2.printMatrix()

    n = Matrix.multiply(m1, m2)
    n.printMatrix()



