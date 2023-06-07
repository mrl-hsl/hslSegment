import numpy as np

# Input dimensions
INPUT_WIDTH = 320
INPUT_HEIGHT = 240

# Output dimensions
OUTPUT_WIDTH = 320
OUTPUT_HEIGHT = 240

color_palette = {
    'background': np.array([0.0, 0.0, 0.0], dtype=np.float32),  # background
    'grass': np.array([0.0, 255.0, 0.0], dtype=np.float32),  # grass
    'line': np.array([0.0, 0.0, 255.0], dtype=np.float32),  # line
    # 'ball': [255, 0, 0], # ball
    # 'goal': [255, 0, 0], # goal
    # 'penalty': [0, 255, 255], # penalty
    # 'circle': [0, 0, 255], # circle
}

MAX_MODELS_TO_KEEP = 5
