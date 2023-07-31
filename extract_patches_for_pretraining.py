import h5py
import openslide
import os
import pandas as pd
import numpy as np
import tqdm

def main(h5_patches_path, test_df_path, extracted_patches_path):

    counter_processed_wsis = 0

    test_df = pd.read_csv(test_df_path)
    test_slide_ids = test_df['slide_id'].tolist()

    counter = 0
    counter_patches = 0
    patch_size = 224
    patch_level = 0

    already_excisting_patches = os.listdir(extracted_patches_path)
    already_excisting_patches = list(map(lambda x: x.split('_')[0], already_excisting_patches))

    for h5_item in os.listdir(h5_patches_path):

        h5_item_name = h5_item[:-3]

        if h5_item_name in test_slide_ids:
            continue

        if h5_item_name in already_excisting_patches:
            continue

        else:
            counter_processed_wsis += 1
            with h5py.File(os.path.join(h5_patches_path, h5_item),'r') as hdf5_file:
                coords = hdf5_file['coords'][:]
                # sample 2000 random values from a numpy array
                if len(coords) > 2000:
                    random_coords = coords[np.random.choice(coords.shape[0], 2000, replace=False), :]
                else:
                    continue
                
                for coord in tqdm.tqdm(random_coords):
                    counter_patches += 1
                    wsi = openslide.open_slide(f"/data/Maack/PANT/ndpi_reduced/NDPI/{h5_item_name}.ndpi")
                    img = wsi.read_region(coord, patch_level, (patch_size, patch_size)).convert('RGB')
                    img.save(os.path.join(extracted_patches_path, f'{h5_item_name}_{coord[0]}_{coord[1]}.png'))
                
                print(f"Processed {counter_processed_wsis} WSIs")
    
    print(f"Processed {counter_processed_wsis} WSIs")
    print(f"Processed {counter_patches} patches")

if __name__ == '__main__':

    extracted_patches_path = "/data/Maack/PANT/CLAM/experiments/002/extracted_patches_for_pretrain"

    h5_patches_path = "/data/Maack/PANT/CLAM/experiments/002/patches"
    
    test_df_path = "/home/Maack/Medulloblastoma/CLAM/dataset_csv/CMB_vs_DN_test.csv"

    main(h5_patches_path, test_df_path, extracted_patches_path)