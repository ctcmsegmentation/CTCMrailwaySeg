import cv2
import PyEMD
import joblib

import numpy as np
from copy import deepcopy
from tqdm import tqdm
from typing import Tuple, Any

from skimage.color import deltaE_ciede2000, rgb2lab, deltaE_cie76
from sklearn.linear_model import LinearRegression
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture


class FeaturesExtractor:
    """
    Extract features from a video where each frame contains tiled subimages:
    top-left: image, top-right: bisenet, bottom-left: RFbisenet, bottom-right: ctcm.

    Parameters
    ----------
    video_storage_path : str
        Path to the video file.
    n_components : int
        Number of GMM components.
    training : bool
        If True, initialize a new GaussianMixture; otherwise load a persisted model.
    """

    def __init__(self, video_storage_path: str, n_components: int = 5, training: bool = False) -> None:
        self.video_storage_path = video_storage_path
        self.n_components = n_components
        self.training = training
        if training:
            self.clf = GaussianMixture(n_components=n_components, random_state=42, max_iter=1000)
        else:
            self.clf = joblib.load("data/gmm_valid.joblib")

    def get_sample_images(self, sampling: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Read video and sample frames uniformly (every `sampling` frames).
        Returns four arrays: images (LAB), bisenet masks, RFbisenet masks, ctcm masks.
        """
        cap = cv2.VideoCapture(self.video_storage_path)
        if not cap.isOpened():
            RuntimeError("Can't open video")
            exit()

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        collected_images = np.zeros((1, 512, 1024, 3))
        collected_bisenet = np.zeros((1, 512, 1024))
        collected_RFbisenet = np.zeros((1, 512, 1024))
        collected_ctcm = np.zeros((1, 512, 1024))

        pbar = tqdm(total=frame_count)
        counter = 0

        while True:
            pbar.n += 1
            pbar.refresh()
            ret, frame = cap.read()
            if ret:
                counter += 1
                if counter % sampling == 0:
                    image = frame[:512, :1024, :]
                    bisenetv2 = frame[:512, 1024:, :]
                    RFbisenetv2 = frame[512:, :1024, :]
                    ctcm = frame[512:, 1024:, :]

                    # convert and threshold as in original
                    image = rgb2lab(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    bisenetv2 = cv2.cvtColor(bisenetv2, cv2.COLOR_BGR2GRAY)
                    bisenetv2 = cv2.threshold(bisenetv2, 127, 255, cv2.THRESH_BINARY)[1] / 255

                    RFbisenetv2 = cv2.cvtColor(RFbisenetv2, cv2.COLOR_BGR2GRAY)
                    RFbisenetv2 = cv2.threshold(RFbisenetv2, 127, 255, cv2.THRESH_BINARY)[1] / 255

                    ctcm = cv2.cvtColor(ctcm, cv2.COLOR_BGR2GRAY)
                    ctcm = cv2.threshold(ctcm, 127, 255, cv2.THRESH_BINARY)[1] / 255
                    ctcm = cv2.erode(ctcm.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=0)

                    collected_images = np.concatenate((collected_images, image[np.newaxis]), axis=0)
                    collected_bisenet = np.concatenate((collected_bisenet, bisenetv2[np.newaxis]), axis=0)
                    collected_RFbisenet = np.concatenate((collected_RFbisenet, RFbisenetv2[np.newaxis]), axis=0)
                    collected_ctcm = np.concatenate((collected_ctcm, ctcm[np.newaxis]), axis=0)

            if pbar.n > pbar.total:
                break

        cap.release()
        return collected_images[1:], collected_bisenet[1:], collected_RFbisenet[1:], collected_ctcm[1:]

    def extract_sample(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Given image (stack of LAB images) and mask (binary) extract:
        - rails: pixels where mask == 1
        - otherwise: pixels where mask == 0
        - surroundings: pixels where enlarged (dilated) mask == 1 excluding mask area

        Returns flattened arrays for each set of pixels.
        """
        enlarged = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        mask_eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3)
        enlarged[mask_eroded == 1] = 0

        mask_flat = np.hstack(mask_eroded)
        enlarged_flat = np.hstack(enlarged)
        image_flat = np.vstack(image)

        imgLAB_rails = deepcopy(image_flat)
        imgLAB_otherwise = deepcopy(image_flat)
        imgLAB_surroundings = deepcopy(image_flat)

        rails = imgLAB_rails[mask_flat == 1]
        otherwise = imgLAB_otherwise[mask_flat == 0]
        surroundings = imgLAB_surroundings[enlarged_flat == 1]

        return rails, otherwise, surroundings

    def train_gmm(self, images: np.ndarray) -> None:
        """
        Train GMM on provided image pixels and persist to 'gmm.joblib'.
        """
        extracted = images.reshape(-1, images.shape[-1])
        self.clf.fit(extracted)
        joblib.dump(self.clf, "gmm.joblib")

    def predict_gmm(self, x: np.ndarray) -> np.ndarray:
        """Predict cluster labels using the GMM classifier."""
        return self.clf.predict(x)

    def get_score(self, image: np.ndarray, mask: np.ndarray) -> Tuple[Any, Any, float, float]:
        """
        Compute a tuple of scores:
        - EMD between rails and otherwise
        - EMD between rails and surroundings
        - mean color variance of rails
        - combined entropy (local angle entropy + global distance entropy)
        """
        rails, otherwise, surroundings = self.extract_sample(image, mask)

        rails_labels = self.predict_gmm(rails)
        otherwise_labels = self.predict_gmm(otherwise)
        surroundings_labels = self.predict_gmm(surroundings)

        rails_histogram = np.histogram(rails_labels, bins=self.n_components)[0].astype(np.float64)
        otherwise_histogram = np.histogram(otherwise_labels, bins=self.n_components)[0].astype(np.float64)
        surroundings_histogram = np.histogram(surroundings_labels, bins=self.n_components)[0].astype(np.float64)

        rails_histogram = (rails_histogram / np.max(rails_histogram))
        otherwise_histogram = (otherwise_histogram / np.max(otherwise_histogram))
        surroundings_histogram = (surroundings_histogram / np.max(surroundings_histogram))

        centroids_distances = np.zeros((self.n_components, self.n_components))
        for i in range(self.n_components):
            for j in range(self.n_components):
                centroids_distances[i, j] = (
                    deltaE_ciede2000(self.clf.means_[i, :], self.clf.means_[j, :])
                    * deltaE_cie76(self.clf.means_[i, :], self.clf.means_[j, :])
                )

        emd_r_o = PyEMD.emd(rails_histogram, otherwise_histogram, centroids_distances)
        emd_r_s = PyEMD.emd(rails_histogram, surroundings_histogram, centroids_distances)

        gc = self.local_angle_entropy(mask) + self.global_distance_entropy(mask)

        return emd_r_o, emd_r_s, self.segments_color_variance(rails), gc

    def segments_color_variance(self, colours: np.ndarray) -> float:
        """
        Compute mean deltaE_ciede2000 distance from the mean color.
        """
        mean_color = np.mean(colours, axis=0)
        distances = [deltaE_ciede2000(mean_color, color) for color in colours]
        variance = float(np.mean(np.array(distances)))
        return variance

    def local_angle_entropy(self, binary_mask: np.ndarray, num_bins: int = 45) -> float:
        """
        Compute local angle entropy from binary mask points.
        Mirrors original algorithm (KD-tree nearest neighbors, angle histogram).
        """
        points = np.argwhere(binary_mask == 1)
        if len(points) < 3:
            raise ValueError("At least three points are required to compute Local Angle Entropy.")

        kdtree = cKDTree(points)
        angles = []

        for _, p_j in enumerate(points):
            distances, neighbors_idx = kdtree.query(p_j, k=3)
            neighbors_idx = neighbors_idx[1:]

            p_i, p_k = points[neighbors_idx[0]], points[neighbors_idx[1]]

            v_ij = p_i - p_j
            v_kj = p_k - p_j

            norm_v_ij = np.linalg.norm(v_ij)
            norm_v_kj = np.linalg.norm(v_kj)

            if norm_v_ij == 0 or norm_v_kj == 0:
                continue

            cos_theta = np.dot(v_ij, v_kj) / (norm_v_ij * norm_v_kj)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            angles.append(theta)

        angles = np.array(angles)
        histogram, bin_edges = np.histogram(angles, bins=num_bins, range=(0, np.pi), density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        probabilities = histogram * bin_width
        probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(probabilities * np.log(probabilities))
        return float(entropy)

    def global_distance_entropy(self, binary_mask: np.ndarray, num_bins: int = 64) -> float:
        """
        Compute global distance entropy:
        - split mask left/right around vertical center
        - fit linear regression x ~ y for each side, compute residuals and entropy
        """
        binary_mask[binary_mask != 1] = 0

        height, width = binary_mask.shape
        x_middle = width // 2

        left_mask = binary_mask[:, :x_middle]
        right_mask = binary_mask[:, x_middle:]

        left_coords = np.argwhere(left_mask > 0)
        right_coords = np.argwhere(right_mask > 0)

        def compute_entropy(coords: np.ndarray) -> Any:
            if len(coords) < 3:
                return None
            y = coords[:, 0]
            x = coords[:, 1]
            model = LinearRegression().fit(y.reshape(-1, 1), x)
            a, b = model.coef_[0], model.intercept_

            predicted_x = a * y + b
            distances = np.abs(x - x_middle)
            predicted_distances = np.abs(predicted_x - x_middle)
            residuals = np.abs(distances - predicted_distances)

            histogram, bin_edges = np.histogram(residuals, bins=num_bins * 2, range=(0, 256), density=True)
            bin_width = bin_edges[1] - bin_edges[0]
            probabilities = histogram * bin_width
            probabilities = probabilities[probabilities > 0]
            return -np.sum(probabilities * np.log(probabilities))

        left_entropy = compute_entropy(left_coords)
        right_entropy = compute_entropy(right_coords)

        if left_entropy is not None and right_entropy is not None:
            return float((left_entropy + right_entropy) / 2)
        if left_entropy is not None:
            return float(left_entropy)
        if right_entropy is not None:
            return float(right_entropy)
        return 0.0

    def get_silhouette(self, data: np.ndarray) -> Any:
        """Return silhouette-like score object (delegates to Silhouette class)."""
        s = Silhouette(self.clf)
        return s.score(data)


class Silhouette:
    """
    Silhouette-like metric based on GMM centroids and color distance measures.
    Implementation preserves original algorithmic steps.
    """

    def __init__(self, clf: GaussianMixture) -> None:
        self._clf = clf

    def _sample_cohesion(self, sample: np.ndarray, data_labels: np.ndarray) -> float:
        centroids = self._clf.means_
        x_label = self._clf.predict(sample.reshape(1, -1))[0]
        x_proba = self._clf.predict_proba(sample.reshape(1, -1))[0]
        agg = deltaE_ciede2000(centroids[x_label], sample) * (1 - x_proba[x_label])
        cardinality = len(np.where(data_labels == x_label)[0])
        cohesion = agg / (cardinality - 1)
        return cohesion

    def _sample_separation(self, sample: np.ndarray, data_labels: np.ndarray) -> float:
        centroids = self._clf.means_
        x_label = self._clf.predict(sample.reshape(1, -1))[0]
        x_proba = self._clf.predict_proba(sample.reshape(1, -1))[0]
        distances = np.apply_along_axis(lambda centroid: deltaE_ciede2000(sample, centroid), 1, centroids)
        mask = np.arange(len(centroids)) != x_label
        scores = distances[mask] * (1 - x_proba[mask]) / np.bincount(data_labels)[mask]
        return np.min(scores)

    def _sample_score(self, sample: np.ndarray, data_labels: np.ndarray) -> float:
        x_label = self._clf.predict(sample.reshape(1, -1))[0]
        score = 0.0
        if len(np.where(data_labels == x_label)[0]) > 1:
            a = self._sample_cohesion(sample, data_labels)
            b = self._sample_separation(sample, data_labels)
            score = (b - a) / max(a, b)
        return float(score)

    def _lab_histogram(self, data: np.ndarray, n_bins_L: int = 20, n_bins_a: int = 20, n_bins_b: int = 20):
        bins_L = np.linspace(0, 100, n_bins_L + 1)
        bins_a = np.linspace(-128, 127, n_bins_a + 1)
        bins_b = np.linspace(-128, 127, n_bins_b + 1)

        hist, edges = np.histogramdd(data, bins=[bins_L, bins_a, bins_b])
        centers_L = (edges[0][:-1] + edges[0][1:]) / 2
        centers_a = (edges[1][:-1] + edges[1][1:]) / 2
        centers_b = (edges[2][:-1] + edges[2][1:]) / 2

        superpixel_dict = {}
        for i, l_bin in enumerate(centers_L):
            for j, a_bin in enumerate(centers_a):
                for k, b_bin in enumerate(centers_b):
                    if hist[i, j, k] > 0:
                        superpixel_dict[(l_bin, a_bin, b_bin)] = hist[i, j, k]

        return superpixel_dict

    def score(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Compute the silhouette-like metric over the provided data.
        Returns a tuple: (weighted_mean_score * 1e7, median_score * 1e7)
        """
        data_flat = data.reshape(-1, data.shape[-1])
        data_labels = self._clf.predict(data_flat)

        lab_hist_dict = self._lab_histogram(data_flat)

        scores = []
        for superpixel, frequency in tqdm(lab_hist_dict.items(), leave=False):
            sample = np.array(superpixel)
            scores.append(self._sample_score(sample, data_labels) * (frequency / data_flat.shape[0]))

        return np.mean(scores) * 1e7, np.median(scores) * 1e7
