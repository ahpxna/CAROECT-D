import cv2
import imageio
import numpy as np

# Parameters
width, height = 400, 400
grid_size = 8
square_size = width // grid_size
duration_sec = 0.5  # Time per frame in seconds

# Create two inverted checkerboard frames using NumPy/OpenCV
def draw_board(invert=False):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(grid_size):
        for j in range(grid_size):
            if (i + j) % 2 == (1 if invert else 0):
                pt1 = (j * square_size, i * square_size)
                pt2 = ((j + 1) * square_size, (i + 1) * square_size)
                cv2.rectangle(img, pt1, pt2, (255, 255, 255), -1)
    # Convert BGR to RGB for correct colors in imageio
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Generate alternating frames
frame1 = draw_board(invert=False)
frame2 = draw_board(invert=True)

# Create a blinking sequence (e.g., 6 flashes)
frames = []
for _ in range(20000):
    frames.append(frame1)
    frames.append(frame2)

# Save frames as an animated GIF
imageio.mimsave('blinking_checkerboard.gif', frames, duration=duration_sec)
print("GIF saved successfully!")
