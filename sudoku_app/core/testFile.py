
import cv2
import numpy as np
import os

from predict_digit import predict_cell

imgPath = "C:/Users/matan/PycharmProjects/SudokuProj/sudoku_app/media/processed_sudokus_page-0001.jpg"
warped_image = cv2.imread(imgPath)
side_length = warped_image.shape[0]
cell_size = side_length // 9
cells = []
digit_grid = []

for row in range(9):
    digit_row = []
    for col in range(9):
        x1 = col * cell_size
        y1 = row * cell_size
        x2 = (col + 1) * cell_size
        y2 = (row + 1) * cell_size
        cell = warped_image[y1:y2, x1:x2]
        # Resize the cell for the CNN (e.g., 28x28 for MNIST-style models)
        if len(cell.shape) == 3:
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        resized_cell = cv2.resize(cell, (50, 50))

        # Apply adaptive thresholding for better digit contrast
        # processed_cell = cv2.adaptiveThreshold(resized_cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                      cv2.THRESH_BINARY_INV, 11, 2)
        digit, confidence = predict_cell(resized_cell)
        # Treat low-confidence predictions as empty cells
        print(f'row={row}, col={col}, pred={digit}')





