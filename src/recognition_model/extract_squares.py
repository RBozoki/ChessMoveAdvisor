def extract_squares(warped_image):
    squares = []
    h, w = warped_image.shape[:2]
    square_h, square_w = h // 8, w // 8
    for i in range(8):
        for j in range(8):
            square = warped_image[i * square_h:(i + 1) * square_h, j * square_w:(j + 1) * square_w]
            squares.append(square)
    return squares