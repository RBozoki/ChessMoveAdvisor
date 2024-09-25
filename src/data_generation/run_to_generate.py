from data_generator import generate_images_from_fen_list
from fen_generator import generate_fen_dataset

dataset = generate_fen_dataset(10)

generate_images_from_fen_list(dataset, "fen_image_associations.csv")
#%%
