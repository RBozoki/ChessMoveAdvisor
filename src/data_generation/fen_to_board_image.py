
from PIL import Image
from fentoboardimage import fenToImage, loadPiecesFolder

def generate_board_image(fen, filename):
    boardImage = fenToImage(
        fen=fen,
        squarelength=100,
        pieceSet=loadPiecesFolder("../../pieces/piece_set_name"),
        darkColor="#D18B47",
        lightColor="#FFCE9E"
    )

    boardImage.save(f"./../../data/{filename}.png", format="PNG")

#%%
