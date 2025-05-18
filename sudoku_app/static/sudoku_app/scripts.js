// Sudoku Grid Input Validation and Data Collection
document.addEventListener("DOMContentLoaded", () => {
    const grid = document.getElementById("sudoku-grid");
    const form = document.getElementById("start-game-form");
    const gridDataInput = document.getElementById("grid-data");

    // Collect grid data on form submit
    form.addEventListener("submit", (event) => {
        const gridData = [];
        const rows = grid.querySelectorAll("tr");

        rows.forEach((row) => {
            const rowData = [];
            row.querySelectorAll("td").forEach((cell) => {
                const value = cell.textContent.trim();
                rowData.push(value === "" ? 0 : parseInt(value));
            });
            gridData.push(rowData);
        });

        // Set the grid data as a JSON string
        gridDataInput.value = JSON.stringify(gridData);
    });

    // Basic digit validation
    grid.addEventListener("input", (event) => {
        const cell = event.target;
        const value = cell.textContent.trim();

        // Allow only digits 1-9
        if (!/^[1-9]$/.test(value)) {
            cell.textContent = "";
            cell.classList.add("error-cell");
        } else {
            cell.classList.remove("error-cell");
        }
    });

    grid.addEventListener("focusout", (event) => {
        const cell = event.target;
        const value = cell.textContent.trim();

        // Remove error if the cell is valid
        if (/^[1-9]$/.test(value)) {
            cell.classList.remove("error-cell");
        }
    });
});


// Solve the Sudoku
function solveSudoku() {
    console.log("SOLVED_GRID:", SOLVED_GRID);  // Check if the solved grid is loaded

    const table = document.getElementById("sudoku-grid");

    for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {
            const cell = table.rows[i].cells[j];
            const value = SOLVED_GRID[i][j];
            const isDetected = DETECTED_GRID[i][j];

            console.log(`Cell [${i}][${j}] -> Detected: ${isDetected}, Value: ${value}`);

            // Only update non-detected cells
            if (!isDetected) {
                cell.textContent = value;
                cell.classList.remove("editable-cell");
                cell.className = "solved-cell";
            }
        }
    }

    console.log("Solve button clicked");
}


// Clear the User-Entered Digits
function clearSolution() {
    const table = document.getElementById("sudoku-grid");

    for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {
            const cell = table.rows[i].cells[j];
            const value = DIGIT_GRID[i][j];

            // Only clear non-detected cells
            if (!DETECTED_GRID[i][j]) {
                if (value === 0) {
                    cell.textContent = "";
                    cell.classList.add("editable-cell");
                } else {
                    cell.textContent = value;
                    cell.classList.remove("editable-cell");
                    cell.classList.add("solved-cell");
                }
            }
            cell.classList.remove("error-cell");
        }
    }
}

// Check the User's Solution
function checkUserSolution() {
    const table = document.getElementById("sudoku-grid");
    const messageContainer = document.getElementById("message-container");
    let isCorrect = true;

    for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {
            const cell = table.rows[i].cells[j];
            const userValue = parseInt(cell.textContent) || 0;
            const correctValue = SOLVED_GRID[i][j];

            // Check if the cell is incorrect
            if (userValue !== correctValue) {
                cell.classList.add("error-cell");
                isCorrect = false;
            } else {
                cell.classList.remove("error-cell");
                cell.classList.add("solved-cell");
            }
        }
    }

    // Display result message
    if (isCorrect) {
        messageContainer.innerHTML = `<div class="success-message">🎉 Your solution is correct! Well done!</div>`;
    } else {
        messageContainer.innerHTML = `<div class="error-message">❌ Your solution has errors. Keep trying!</div>`;
    }

    // Automatically hide the message after 5 seconds
    setTimeout(() => {
        messageContainer.innerHTML = "";
    }, 5000);
}
