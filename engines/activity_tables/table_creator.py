import pygame
import sys
import numpy as np
import os

# Configuration
piece = 'king'  # Change this to one of ['knight', 'bishop', 'pawn', 'queen', 'rook','king'] as needed
SELECT_MULTIPLE = False
filename = f"{piece}_activity_table.npy"

# Constants
ROWS, COLS = 8, 8
CELL_SIZE = 60
MARGIN = 2
GRID_WIDTH = COLS * (CELL_SIZE + MARGIN) + MARGIN
GRID_HEIGHT = ROWS * (CELL_SIZE + MARGIN) + MARGIN
BUTTON_HEIGHT = 50
TOP_MARGIN = 50  # space at the top for the piece text
WINDOW_HEIGHT = GRID_HEIGHT + BUTTON_HEIGHT + TOP_MARGIN
WINDOW_WIDTH = GRID_WIDTH
FONT_SIZE = 24

last_hovered_cell = None

pygame.init()
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Activity Table Editor")
font = pygame.font.SysFont(None, FONT_SIZE)

# Attempt to load existing table if it exists
if os.path.exists(filename):
    loaded_arr = np.load(filename)
    # Ensure it matches the expected shape
    if loaded_arr.shape == (ROWS, COLS):
        board_values = [[str(int(loaded_arr[r, c])) for c in range(COLS)] for r in range(ROWS)]
    else:
        print("Loaded file shape does not match expected 8x8. Starting with defaults.")
        board_values = [["0" for _ in range(COLS)] for _ in range(ROWS)]
else:
    board_values = [["0" for _ in range(COLS)] for _ in range(ROWS)]

# Track selected cells
selected_cells = set()

# For dragging selection
selecting = False

def get_cell_from_pos(x, y):
    # Adjust y by subtracting TOP_MARGIN before identifying the cell
    adjusted_y = y - TOP_MARGIN

    if 0 <= adjusted_y < GRID_HEIGHT:  # Check within the adjusted grid height
        col = (x - MARGIN) // (CELL_SIZE + MARGIN)
        row = (adjusted_y - MARGIN) // (CELL_SIZE + MARGIN)
        if 0 <= row < ROWS and 0 <= col < COLS:
            return (row, col)
    return None

def value_to_color(value, min_val=0, max_val=100):
    # Clamp value to [min_val, max_val]
    value = max(min_val, min(value, max_val))
    # Normalize to 0.0 - 1.0
    t = (value - min_val) / (max_val - min_val)
    # Interpolate between white (255,255,255) at t=0 and green (0,255,0) at t=1
    r = int(255 * (1 - t))     # from 255 down to 0
    g = 255                    # stays 255 at all times
    b = int(255 * (1 - t))     # from 255 down to 0
    return (r, g, b)

def draw_grid():
    window.fill((220, 220, 220))
    for r in range(ROWS):
        for c in range(COLS):
            x = MARGIN + c * (CELL_SIZE + MARGIN)
            y = TOP_MARGIN + MARGIN + r * (CELL_SIZE + MARGIN)
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

            # Convert cell value to float for coloring
            try:
                cell_value = int(board_values[r][c])
            except ValueError:
                cell_value = 0
            
            # Get color based on value
            base_color = value_to_color(float(cell_value))
            
            # If cell is selected, we could modify the color or just leave it as is
            # For example, make it slightly brighter if selected:
            if (r, c) in selected_cells:
                # # Make it lighter by mixing with white
                # base_color = (
                #     min(base_color[0] + 40, 255),
                #     min(base_color[1] + 40, 255),
                #     min(base_color[2] + 40, 255)
                # )
                base_color = (255, 255, 200)

            pygame.draw.rect(window, base_color, rect)

            # Draw text value on top
            text_surf = font.render(str(cell_value), True, (0, 0, 0))
            text_rect = text_surf.get_rect(center=rect.center)
            window.blit(text_surf, text_rect)

    # Draw piece name text and buttons as before
    piece_text = font.render(f"Editing: {piece.capitalize()} Activity Table", True, (0, 0, 0))
    window.blit(piece_text, (10, 10))

    clear_button_rect = pygame.Rect(50, TOP_MARGIN + GRID_HEIGHT + (BUTTON_HEIGHT - 30)//2, 120, 30)
    submit_button_rect = pygame.Rect(WINDOW_WIDTH - 170, TOP_MARGIN + GRID_HEIGHT + (BUTTON_HEIGHT - 30)//2, 120, 30)

    pygame.draw.rect(window, (180, 70, 70), clear_button_rect)
    clear_text = font.render("Clear Selection", True, (255, 255, 255))
    window.blit(clear_text, clear_text.get_rect(center=clear_button_rect.center))

    pygame.draw.rect(window, (70, 130, 180), submit_button_rect)
    btn_text = font.render("Submit", True, (255, 255, 255))
    window.blit(btn_text, btn_text.get_rect(center=submit_button_rect.center))

    return clear_button_rect, submit_button_rect

def save_to_file():
    # Convert board_values to a numpy array of floats
    arr = np.zeros((ROWS, COLS), dtype=np.int32)
    for r in range(ROWS):
        for c in range(COLS):
            try:
                arr[r, c] = int(board_values[r][c])
            except ValueError:
                arr[r, c] = 0
    np.save(filename, arr)
    print(f"Saved activity table to {filename}")

def backspace_selected_cells():
    for (r, c) in selected_cells:
        if len(board_values[r][c]) > 0:
            board_values[r][c] = board_values[r][c][:-1]
        if board_values[r][c] == "":
            board_values[r][c] = "0"

running = True
while running:
    clear_button_rect, submit_button_rect = draw_grid()
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # left click
                mx, my = event.pos
                # Check for button clicks
                if clear_button_rect.collidepoint(mx, my):
                    selected_cells.clear()
                elif submit_button_rect.collidepoint(mx, my):
                    save_to_file()
                    running = False
                else:
                    # Check if on board
                    cell = get_cell_from_pos(mx, my)
                    if cell:
                        if SELECT_MULTIPLE:
                            # Toggle selection of the clicked cell
                            if cell in selected_cells:
                                selected_cells.remove(cell)
                            else:
                                selected_cells.add(cell)
                            
                            # If you still want the ability to drag and add cells,
                            # set selecting to True so that if the user moves the mouse 
                            # while holding the button, additional cells can be added.
                            selecting = True
                            last_hovered_cell = cell
                        else:
                            selected_cells = set()
                            selected_cells.add(cell)
                            selecting = True
                            last_hovered_cell = cell
                    else:
                        # Clicked outside the board, do nothing
                        pass

        elif event.type == pygame.MOUSEMOTION:
            if selecting:
                mx, my = event.pos
                current_cell = get_cell_from_pos(mx, my)
                # Only toggle if we moved to a *different* cell
                if current_cell and current_cell != last_hovered_cell:
                    # Toggle this cell
                    if current_cell in selected_cells:
                        selected_cells.remove(current_cell)
                    else:
                        selected_cells.add(current_cell)
                    last_hovered_cell = current_cell

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and selecting:
                selecting = False
                last_hovered_cell = None

        elif event.type == pygame.KEYDOWN:
            if selected_cells:
                if event.key == pygame.K_BACKSPACE:
                    backspace_selected_cells()
                elif event.key == pygame.K_MINUS:
                    # If all selected are "0", allow minus
                    all_zero = all(board_values[r][c] == "0" for (r, c) in selected_cells)
                    if all_zero:
                        for (r, c) in selected_cells:
                            board_values[r][c] = "-"
                elif event.unicode.isdigit():
                    # Digit typed
                    for (r, c) in selected_cells:
                        current = board_values[r][c]
                        if current == "0":
                            board_values[r][c] = event.unicode
                        else:
                            board_values[r][c] = current + event.unicode

pygame.quit()
sys.exit()