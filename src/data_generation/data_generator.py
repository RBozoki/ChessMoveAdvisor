import csv

from fen_to_board_image import generate_board_image

def generate_images_from_fen_list(fen_list, csv_filename):
    with open(f"./../../data/{csv_filename}", mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(["filename", "fen"])

        for i, fen in enumerate(fen_list):
            filename = f"chess_position_{i}"

            generate_board_image(fen, filename)

            writer.writerow([filename, fen])

