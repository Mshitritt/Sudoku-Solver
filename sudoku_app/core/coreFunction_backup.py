import cv2
import numpy as np
import os
from .predict_digit import predict_cell

MEDIA_DIR = 'sudoku_app/media'

def extract_sudoku_grid(input_path, output_path, contour_path, cell_dir, img_name):
    # Read the uploaded image
    image = cv2.imread(input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # save preprocess 1 -
    preProcess_1 = os.path.join(MEDIA_DIR, f'preProcess_1_{img_name}.png')
    cv2.imwrite(preProcess_1, blurred)


    # **Improved Edge Detection**
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # save preprocess thresh
    preProcess_2 = os.path.join(MEDIA_DIR, f'thresh_2_{img_name}.png')
    cv2.imwrite(preProcess_2, thresh)

    # **Improvement:** Morphological operations to clean up the grid
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph_img = os.path.join(MEDIA_DIR, f'morphological_{img_name}.png')
    cv2.imwrite(morph_img, closed)

    # **Improvement:** Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.erode(closed, kernel, iterations=1)
    lessNoise = os.path.join(MEDIA_DIR, f'lessNoise_{img_name}.png')
    cv2.imwrite(lessNoise, cleaned)

    # **Improved Contour Detection**
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find the grid by checking for the largest 4-sided contour
    grid_contour = None
    area = 0
    for contour in contours:
        # Approximate the contour to reduce points
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check for a 4-sided polygon
        if len(approx) == 4:
            # Check for a roughly square shape
            points = np.array([point[0] for point in approx], dtype='float32')
            width1 = np.linalg.norm(points[0] - points[1])
            width2 = np.linalg.norm(points[2] - points[3])
            height1 = np.linalg.norm(points[0] - points[3])
            height2 = np.linalg.norm(points[1] - points[2])

            aspect_ratio = max(width1, width2) / max(height1, height2)
            area = cv2.contourArea(contour)
            edge_density = cv2.countNonZero(thresh) / cv2.contourArea(contour)
            if 0.6 <= aspect_ratio <= 2 and area > 50 and edge_density > 0.3:
                grid_contour = approx
                break

    # **Draw Detected Grid Contour**
    imgContour = image.copy()
    if grid_contour is not None:
        cv2.drawContours(imgContour, [grid_contour], -1, (0, 255, 0), 10)
        cv2.imwrite(contour_path, imgContour)

        # **Improved Corner Sorting**
        points = np.array([point[0] for point in grid_contour], dtype='float32')
        s = points.sum(axis=1)
        diff = np.diff(points, axis=1)

        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = points[np.argmin(s)]  # top-left
        rect[2] = points[np.argmax(s)]  # bottom-right
        rect[1] = points[np.argmin(diff)]  # top-right
        rect[3] = points[np.argmax(diff)]  # bottom-left

        # Calculate the maximum side length for the target square
        side_length = int(max(
            np.linalg.norm(rect[0] - rect[1]),
            np.linalg.norm(rect[1] - rect[2]),
            np.linalg.norm(rect[2] - rect[3]),
            np.linalg.norm(rect[3] - rect[0])
        ))

        # Define the target square
        target_square = np.array([
            [0, 0],
            [side_length - 1, 0],
            [side_length - 1, side_length - 1],
            [0, side_length - 1]
        ], dtype='float32')

        # Apply perspective transformation
        matrix = cv2.getPerspectiveTransform(rect, target_square)
        warped = cv2.warpPerspective(image, matrix, (side_length, side_length))

        # Save the processed image
        cv2.imwrite(output_path, warped)


        print(f"Grid extracted and saved to {output_path}")
        return extract_cells(warped, cell_dir)

    else:
        print("No valid Sudoku grid found.")
        return [0, 0, 0]


def extract_cells(warped_image, cell_dir):
    os.makedirs(cell_dir, exist_ok=True)
    side_length = warped_image.shape[0]
    cell_size = side_length // 9
    cells = []
    digit_grid = []
    detected_grid = []
    confidence_score = []

    for row in range(9):
        digit_row = []
        detected_row = []
        confidence_score_row = []
        for col in range(9):
            x1 = col * cell_size
            y1 = row * cell_size
            x2 = (col + 1) * cell_size
            y2 = (row + 1) * cell_size
            cell = warped_image[y1:y2, x1:x2]
            # Resize the cell for the CNN
            if len(cell.shape) == 3:
                cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            resized_cell = cv2.resize(cell, (50, 50))

            # Apply adaptive thresholding for better digit contrast
            # processed_cell = cv2.adaptiveThreshold(resized_cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            #                                      cv2.THRESH_BINARY_INV, 11, 2)
            digit, confidence = predict_cell(resized_cell)
            # Treat low-confidence predictions as empty cells
            if confidence < 0.9:
                digit = -1
                detected_row.append(True)
            else:
                if digit == 0:
                    detected_row.append(False)
                else:
                    detected_row.append(True)

            confidence_score_row.append(round(confidence, 3))
            digit_row.append(digit)
            # Save the cell for reference
            cell_path = os.path.join(cell_dir, f"cell_{row}_{col}.png")
            cv2.imwrite(cell_path, resized_cell)

            # Save the cell image
            cell_path = os.path.join(cell_dir, f"cell_{row}_{col}.png")
            cv2.imwrite(cell_path, cell)
            cells.append(cell_path)

        digit_grid.append(digit_row)
        detected_grid.append(detected_row)
        confidence_score.append(confidence_score_row)
        print(f'******* row = {row}, digits row = {digit_row}')

    print(f"Cells saved to {cell_dir}")
    return digit_grid, detected_grid, confidence_score


from scipy.interpolate import griddata

def correct_curved_grid(warped):
    # Convert to grayscale
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Find the grid lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=40, maxLineGap=20)
    if lines is None:
        print("No grid lines found.")
        return warped

    # Collect line endpoints
    points = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            points.append([x1, y1])
            points.append([x2, y2])

    # Convert to NumPy array
    points = np.array(points)

    # Get the bounding rectangle for the grid
    rect = cv2.boundingRect(points)

    # Generate the target grid (perfect 9x9)
    target_points = []
    for i in range(10):
        for j in range(10):
            target_points.append([
                rect[0] + (rect[2] // 9) * j,
                rect[1] + (rect[3] // 9) * i
            ])
    target_points = np.array(target_points, dtype="float32")

    # Interpolate the points using TPS
    grid_x, grid_y = np.meshgrid(
        np.linspace(rect[0], rect[0] + rect[2], 9),
        np.linspace(rect[1], rect[1] + rect[3], 9)
    )
    mapped_x = griddata(points, points[:, 0], (grid_x, grid_y), method='linear')
    mapped_y = griddata(points, points[:, 1], (grid_x, grid_y), method='linear')

    # Apply the TPS transformation
    map_x = np.float32(mapped_x)
    map_y = np.float32(mapped_y)
    corrected = cv2.remap(warped, map_x, map_y, cv2.INTER_LINEAR)

    return corrected


def solve_sudoku(board):
    def is_valid(board, row, col, num):
        # Check row
        if num in board[row]:
            return False
        # Check column
        if num in [board[r][col] for r in range(9)]:
            return False
        # Check 3x3 square
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if board[r][c] == num:
                    return False
        return True

    # Find the first empty cell
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku(board):
                            return True
                        board[row][col] = 0
                return False
    return True



