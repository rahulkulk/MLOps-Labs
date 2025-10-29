#Functions changed: build_save_model and load_model_elbow
#README.md file has the description of the changes done

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from kneed import KneeLocator
import pickle
import os
import base64

def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    print("We are here")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    serialized_data = pickle.dumps(df)                    # bytes
    return base64.b64encode(serialized_data).decode("ascii")  # JSON-safe string

def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data.
    """
    # decode -> bytes -> DataFrame
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)

    # bytes -> base64 string for XCom
    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return base64.b64encode(clustering_serialized_data).decode("ascii")

def build_save_model(data_b64: str, filename: str, n_clusters_range=(1, 50), n_components=2):
    """
    Builds a MiniBatchKMeans model on PCA-transformed preprocessed data and saves both the PCA and model.
    
    Args:
        data_b64 (str): Base64-encoded preprocessed data.
        filename (str): Filename to save the model and PCA object.
        n_clusters_range (tuple): Range of cluster numbers for SSE calculation (default 1-50).
        n_components (int): Number of PCA components (default 2).

    Returns:
        list: SSE values for each number of clusters (JSON-serializable).
    """
    import numpy as np

    # Decode base64 -> bytes -> numpy array
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    # Apply PCA
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df)

    # Initialize MiniBatchKMeans parameters
    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
    sse = []

    # Compute SSE for range of clusters
    for k in range(n_clusters_range[0], n_clusters_range[1]):
        kmeans = MiniBatchKMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df_pca)
        sse.append(kmeans.inertia_)

    # Save both PCA and last-fitted MiniBatchKMeans
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump({"pca": pca, "kmeans": kmeans}, f)

    return sse

def load_model_elbow(filename: str, sse: list):
    """
    Loads the saved PCA + MiniBatchKMeans model and predicts cluster for test data.
    """
    import pandas as pd
    import pickle
    import os
    from kneed import KneeLocator

    # Load the saved PCA + MiniBatchKMeans
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    model_dict = pickle.load(open(output_path, "rb"))
    pca = model_dict["pca"]
    kmeans = model_dict["kmeans"]

    # Load test data
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))

    # Apply same PCA transformation
    df_pca = pca.transform(df)

    # Predict clusters
    pred = kmeans.predict(df_pca)[0]

    # Elbow info
    kl = KneeLocator(range(1, 50), sse, curve="convex", direction="decreasing")
    print(f"Optimal no. of clusters: {kl.elbow}")

    # Ensure JSON-safe return
    try:
        return int(pred)
    except Exception:
        return pred.item() if hasattr(pred, "item") else pred
