from django.shortcuts import render
from .forms import SudokuImageForm
from .core.core_functions import extract_sudoku_grid, solve_sudoku
import os
import json
from django.http import JsonResponse

EXTRACTED_DATA = {}

MEDIA_DIR = 'sudoku_app/media'

def save_uploaded_file(file):
    image_path = os.path.join(MEDIA_DIR, file.name)
    with open(image_path, 'wb+') as f:
        for chunk in file.chunks():
            f.write(chunk)
    return image_path

def upload_page(request):
    image_url = None
    processed_image_url = None
    contour_image_url = None
    cell_images = []
    message = ""
    grid_detected = False
    confidence = []
    preProcess_1, preProcess_2 = None, None
    morph_path, canny_path = None, None
    allContours_path = None


    if request.method == 'POST' and 'upload' in request.POST:
        form = SudokuImageForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded file
            image_file = form.cleaned_data['sudoku_image']
            image_path = save_uploaded_file(image_file)

            # Generate paths for processed images
            base_name = os.path.splitext(image_file.name)[0]
            preProcess_1 = f'/media/preprocess_1_{base_name}.png'   # Gray
            canny_path = f'/media/canny_{base_name}.png'        # Canny
            allContours_path = f'/media/allContours_{base_name}.png'
            contour_image_name = f"contour_{base_name}.png"
            cell_dir = os.path.join(MEDIA_DIR, f"cells_{base_name}")

            processed_image_name = f"processed_{base_name}.png"

            # Ensure the cell directory exists
            if not os.path.exists(cell_dir):
                os.makedirs(cell_dir)

            processed_image_path = os.path.join(MEDIA_DIR, processed_image_name)
            contour_image_path = os.path.join(MEDIA_DIR, contour_image_name)

            # Run the grid detection and digit prediction
            digit_grid, detected_grid, confidence = extract_sudoku_grid(
                image_path, processed_image_path, contour_image_path, cell_dir, base_name
            )

            # Validate the extracted grid
            if digit_grid:

                # Save extracted data for the game page
                EXTRACTED_DATA['digit_grid'] = digit_grid
                EXTRACTED_DATA['detected_grid'] = detected_grid

                EXTRACTED_DATA['preProcess_1'] = preProcess_1
                EXTRACTED_DATA['preProcess_2'] = preProcess_2

                EXTRACTED_DATA['allContours_path'] = morph_path
                EXTRACTED_DATA['canny_path'] = canny_path



                # Set image URLs for display
                image_url = f'/media/{image_file.name}'
                processed_image_url = f'/media/{processed_image_name}'
                contour_image_url = f'/media/{contour_image_name}'
                grid_detected = True
                message = "Grid Detected"

                # Collect cell images for the table
                for row in range(9):
                    cell_row = []
                    for col in range(9):
                        cell_image_name = f"cell_{row}_{col}.png"
                        cell_image_url = f"/media/cells_{base_name}/{cell_image_name}"
                        cell_row.append(cell_image_url)
                    cell_images.append(cell_row)

                EXTRACTED_DATA['cell_images'] = cell_images
                EXTRACTED_DATA['confidence'] = confidence
                EXTRACTED_DATA['board_path'] = processed_image_url

            else:

                # Set image URLs for display
                image_url = f'/media/{image_file.name}'
                contour_image_url = f'/media/{contour_image_name}'
                message = "Error: The uploaded image does not contain a valid Sudoku grid."

    else:
        form = SudokuImageForm()

    return render(request, 'sudoku_app/upload.html', {
        'form': form,
        'image_url': image_url,
        'processed_image_url': processed_image_url,
        'contour_image_url': contour_image_url,
        'message': message,
        'grid_detected': grid_detected,
        'cell_images': cell_images,
        'confidence': confidence,
        'preProcess_1_path': preProcess_1,
        'allContours_path': allContours_path,
        'canny_path': canny_path

    })


def cnn_workflow(request):

    samples_dir = 'sudoku_app/media/reports_images/samples'
    sample_dir_final = '/media/reports_images/samples'
    samples = {}

    # Loop through each digit folder (0-9)
    for digit in range(10):
        digit_dir = os.path.join(samples_dir, str(digit))
        if os.path.isdir(digit_dir):
            samples[digit] = []
            for img_file in sorted(os.listdir(digit_dir))[:10]:
                # Use forward slashes for client-side compatibility
                img_path = f"{sample_dir_final.replace('\\', '/')}/{str(digit)}/{img_file}"
                samples[digit].append(img_path)

    EXTRACTED_DATA['samples'] = samples
    class_report_path = 'sudoku_app/staff/classification_report.txt'
    # Read the classification report
    if os.path.exists(class_report_path):
        with open(class_report_path, 'r', encoding='utf-8') as f:
            classification_report = f.read()
            # Replace any problematic characters
            classification_report = classification_report.replace('<', '&lt;').replace('>', '&gt;')

    else:
        classification_report = "Classification report not found."

    EXTRACTED_DATA['classification_report'] = classification_report

    context = {
        "board_path":  EXTRACTED_DATA['board_path'],
        "cell_images": EXTRACTED_DATA['cell_images'],  # Replace with your actual cell images variable
        "confidence": EXTRACTED_DATA['confidence'],
        'digit_grid': EXTRACTED_DATA.get('digit_grid', []),
        # Staff and Reports paths
        "samples": samples,
        "data_dist_path": f'/media/reports_images/digit_distribution.png',
        "train_loss_path": f'/media/reports_images/training_loss_dark.png',
        "confMatrix_path": f'/media/reports_images/confusion_matrix.png',
        "classification_report": EXTRACTED_DATA['classification_report']
    }
    return render(request, "sudoku_app/cnn_workflow.html", context)


def game_page(request):
    digit_grid = EXTRACTED_DATA.get("digit_grid", [])
    detected_grid = EXTRACTED_DATA.get("detected_grid", [])

    # Copy the digit grid to solve it
    solved_grid = [row.copy() for row in digit_grid]
    solve_sudoku(solved_grid)

    context = {
        'digit_grid': EXTRACTED_DATA.get('digit_grid', []),
        'detected_grid': EXTRACTED_DATA.get('detected_grid', []),
        "grid_data_json": json.dumps(digit_grid),
        "detected_grid_json": json.dumps(detected_grid),
        "solved_grid_json": json.dumps(solved_grid),

    }

    return render(request, "sudoku_app/game_page.html", context)



def home_old(request):
    image_url = None
    processed_image_url = None
    digit_grid = []
    detected_grid = []

    if request.method == 'POST' and 'upload' in request.POST:
        form = SudokuImageForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded file
            image_file = form.cleaned_data['sudoku_image']
            image_path = os.path.join('sudoku_app/media', image_file.name)
            processed_image_name = f"processed_{image_file.name}"
            processed_image_path = os.path.join('sudoku_app/media', processed_image_name)
            cell_dir = os.path.join('sudoku_app/media', f"cells_{image_file.name}")

            with open(image_path, 'wb+') as f:
                for chunk in image_file.chunks():
                    f.write(chunk)

            # Run the grid detection and digit prediction
            digit_grid, detected_grid = extract_sudoku_grid(image_path, processed_image_path, cell_dir)

            # Store the image URLs
            image_url = f'/media/{image_file.name}'
            processed_image_url = f'/media/{processed_image_name}'

    else:
        form = SudokuImageForm()

    return render(request, 'sudoku_app/index.html', {
        'form': form,
        'image_url': image_url,
        'processed_image_url': processed_image_url,
        'digit_grid': digit_grid,
        'detected_grid': detected_grid,
        'grid_json': json.dumps(digit_grid) if digit_grid else "",
        'detected_json': json.dumps(detected_grid) if detected_grid else ""
    })


def solve_grid(request):
    if request.method == 'POST':
        digit_grid = json.loads(request.POST.get('digit_grid'))
        detected_grid = json.loads(request.POST.get('detected_grid'))

        # Solve the grid
        solve_sudoku(digit_grid)

        return JsonResponse({"digit_grid": digit_grid})

    return JsonResponse({"error": "Invalid request"}, status=400)

