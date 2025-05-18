from django import forms

class SudokuImageForm(forms.Form):
    sudoku_image = forms.ImageField(label="Upload Sudoku Image")
