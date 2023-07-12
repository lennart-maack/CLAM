from sklearn import cluster
import h5py
import numpy as np
import cv2


def create_clustered_overlay(h5_features_file, n_clusters, tile_size, slide_size, output_path):

    k_means = cluster.KMeans(n_clusters=n_clusters, random_state=0)

    with h5py.File(h5_features_file, 'r') as f:
        features = f['features'][:]
        coordinates = f['coords'][:]
    
    print(features.shape)

    k_means = k_means.fit(features)

    labels = k_means.labels_

    # possible colors to choose from
    colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [0, 255, 255], [255, 0, 255], [0, 0, 0], [255, 255, 255]]

    img = np.ones(slide_size, dtype=np.uint8)

    for i, (x, y) in enumerate(coordinates):
        x_start=int(x/64)
        x_end=x_start+tile_size

        y_start=int(y/64)
        y_end=y_start+tile_size

        # Make the region of interest intransparent
        img[y_start:y_end, x_start:x_end, 3] = np.ones((tile_size,tile_size), dtype=np.uint8) * 255
        img[y_start:y_end, x_start:x_end, 0] = np.ones((tile_size,tile_size), dtype=np.uint8) * colors[labels[i]][0]
        img[y_start:y_end, x_start:x_end, 1] = np.ones((tile_size,tile_size), dtype=np.uint8) * colors[labels[i]][1]
        img[y_start:y_end, x_start:x_end, 2] = np.ones((tile_size,tile_size), dtype=np.uint8) * colors[labels[i]][2]


    # export img as png
    cv2.imwrite(output_path, img)

    # # export labels as h5 file
    # with h5py.File(h5_features_file_new, 'w') as f:
    #     f.create_dataset('labels', data=labels)
    #     f.create_dataset('coords', data=coordinates)


if __name__ == "__main__":

    h5_features_file = "/home/Maack/Medulloblastoma/example/features/h5_files/33-79-II-2018-04-26-19.45.48.h5"


    n_clusters = 8
    tile_size = int(256/64)
    slide_size = (1316, 2496, 4)

    for n_clusters in range(2, n_clusters):
        output_path = f"/home/Maack/Medulloblastoma/example/{n_clusters}clustered_coordinates.png"
        create_clustered_overlay(h5_features_file, n_clusters, tile_size, slide_size, output_path)


