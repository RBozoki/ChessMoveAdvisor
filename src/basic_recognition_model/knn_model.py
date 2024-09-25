from sklearn.neighbors import KNeighborsClassifier

def train_knn(pieces, labels):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(pieces, labels)
    return knn


def predict_with_threshold(model, image_vector, max_empty_distance, n_neighbors=3):
    image_vector = image_vector.reshape(1, -1)
    distances, indices = model.kneighbors(image_vector, n_neighbors=n_neighbors)

    nearest_label = model.predict(image_vector)[0]
    nearest_distance = distances[0][0]

    if nearest_label == "Empty" and nearest_distance > max_empty_distance:
        for i, index in enumerate(indices[0]):
            label = model.classes_[model._y[index]]
            if label != "Empty":
                return label, distances[0][i]

    return nearest_label, nearest_distance