
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import cv2
import gc


class PRP_PIL:
    def __init__(self,
                 patch_size=6, stride=1, projection_dim=256,
                 n_hidden=2048, activation='tanh',
                 use_multi_scale=False,
                 patch_sizes=None, strides=None, projection_dims=None,
                 use_advanced_features=True,
                 use_attention=True,
                 use_random_projection=True,
                 use_pca=True,
                 use_gabor=False,
                 use_rff=False,
                 use_binary_hashing=False,
                 use_graph_laplacian=False,
                 use_color_hist=False,
                 use_gradients=True,
                 use_feature_standardization=False,
                 regularization=1e-5,
                 graph_lambda=1e-3,
                 rff_dim=128,
                 rff_gamma=0.1,
                 hashing_dim=128,
                 random_state=42,
                 batch_size=200,
                 attention_temperature=1.0):

        # ----- Basic parameters -----
        self.patch_size = patch_size
        self.stride = stride
        self.projection_dim = projection_dim
        self.n_hidden = n_hidden
        self.activation = activation

        # ----- Multi-scale patches -----
        self.use_multi_scale = use_multi_scale
        if use_multi_scale:
            self.patch_sizes = patch_sizes or [4, 8, 12]
            self.strides = strides or [2, 4, 6]
            self.projection_dims = projection_dims or [128, 256, 384]
        else:
            self.patch_sizes = [patch_size]
            self.strides = [stride]
            self.projection_dims = [projection_dim]

        # ----- Feature toggles -----
        self.use_advanced_features = use_advanced_features
        self.use_attention = use_attention
        self.use_random_projection = use_random_projection
        self.use_pca = use_pca
        self.use_gabor = use_gabor
        self.use_rff = use_rff
        self.use_binary_hashing = use_binary_hashing
        self.use_graph_laplacian = use_graph_laplacian
        self.use_color_hist = use_color_hist
        self.use_gradients = use_gradients
        self.use_feature_standardization = use_feature_standardization

        # ----- Regularization / RFF / hashing -----
        self.regularization = regularization
        self.graph_lambda = graph_lambda
        self.rff_dim = rff_dim
        self.rff_gamma = rff_gamma
        self.hashing_dim = hashing_dim

        # ----- Bookkeeping -----
        self.random_state = random_state
        self.batch_size = batch_size
        self.attention_temperature = attention_temperature
        np.random.seed(random_state)
        self.rng = np.random.RandomState(random_state)

        # ----- Learned parameters -----
        self.R_list = []
        self.R_rff = None
        self.b_rff = None
        self.R_hash_dict = {}
        self.gabor_filters = []
        self.W = None
        self.b = None
        self.beta = None
        self.pca = None
        self._pca_fitted = False
        self._feat_mean = None
        self._feat_std = None
        self.encoder = None
        self.n_classes = None
        self.L_graph = None

    # ------------------------------------------------------------------
    def _apply_activation(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
        return np.tanh(x)

    # ------------------------------------------------------------------
    def _extract_patches_at_scale(self, image, patch_size, stride):
        H, W, C = image.shape
        patches = []
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                patch = image[i:i+patch_size, j:j+patch_size, :]
                patches.append(patch.flatten())
        if not patches:
            return np.zeros((1, patch_size * patch_size * C),
                            dtype=np.float32)
        return np.array(patches, dtype=np.float32)

    # ------------------------------------------------------------------
    def _create_gabor_filters(self, num_filters=6):
        filters = []
        for theta in np.linspace(0, np.pi, 4):
            for sigma in [1, 2]:
                kernel = cv2.getGaborKernel(
                    (7, 7), sigma, theta, 8, 0.5, 0,
                    ktype=cv2.CV_32F)
                filters.append(kernel)
        return filters[:num_filters]

    def _gabor_features(self, img):
        feats = []
        gray = img.mean(axis=-1) if img.ndim == 3 else img
        for k in self.gabor_filters:
            resp = cv2.filter2D(gray.astype(np.float32),
                                cv2.CV_32F, k)
            feats.extend([float(resp.mean()), float(resp.std()),
                          float(np.abs(resp).max())])
        return np.array(feats, dtype=np.float32)

    # ------------------------------------------------------------------
    def _advanced_features(self, img):
        feats = [float(np.mean(img)), float(np.std(img))]
        if self.use_advanced_features:
            feats.append(float(np.median(img)))
        if self.use_gradients:
            gray = img.mean(axis=-1)
            gx = np.gradient(gray, axis=1)
            gy = np.gradient(gray, axis=0)
            mag = np.sqrt(gx ** 2 + gy ** 2)
            feats.extend([float(mag.mean()), float(mag.std())])
        if self.use_color_hist:
            for c in range(img.shape[-1]):
                feats.append(float(np.percentile(img[..., c], 90)))
        return np.array(feats, dtype=np.float32)

    # ------------------------------------------------------------------
    def _attention_pooling(self, Z):
        P, L = Z.shape
        if P == 0:
            return np.zeros(L, dtype=np.float32), np.zeros(L, dtype=np.float32)
        if not self.use_attention:
            return Z.max(axis=0), Z.mean(axis=0)
        var = Z.var(axis=1)
        v_mean = var.mean()
        v_std = var.std() + 1e-8
        logits = (var - v_mean) / v_std / self.attention_temperature
        logits = logits - logits.max()
        attn = np.exp(logits) / (np.exp(logits).sum() + 1e-12)
        f_max = Z.max(axis=0)
        f_mean = (Z * attn[:, None]).sum(axis=0)
        return f_max, f_mean

    # ------------------------------------------------------------------
    def _features_one_image(self, img):
        parts = []

        for scale_idx, (ps, st) in enumerate(
                zip(self.patch_sizes, self.strides)):
            patches = self._extract_patches_at_scale(img, ps, st)
            if patches.shape[0] == 0:
                continue

            if self.use_random_projection:
                Z = np.tanh(patches @ self.R_list[scale_idx])
            else:
                Z = np.tanh(patches)

            if self.use_binary_hashing:
                R_h = self.R_hash_dict[Z.shape[1]]
                Z_hash = np.sign(Z @ R_h)
                Z = np.concatenate([Z, Z_hash], axis=1)

            f_max, f_mean = self._attention_pooling(Z)
            parts.append(f_max)
            parts.append(f_mean)

        if self.use_rff:
            flat = img.flatten().astype(np.float32)
            proj = flat @ self.R_rff + self.b_rff
            rff = np.concatenate([np.cos(proj), np.sin(proj)], axis=0)
            parts.append(rff.astype(np.float32))

        if self.use_gabor and self.gabor_filters:
            parts.append(self._gabor_features(img))

        parts.append(self._advanced_features(img))

        return np.concatenate(parts)

    # ------------------------------------------------------------------
    def _create_features_batch(self, X_batch):
        return np.stack([self._features_one_image(img)
                         for img in X_batch]).astype(np.float32)

    def _process_in_batches(self, X, process_func):
        out = []
        for i in tqdm(range(0, len(X), self.batch_size),
                      desc="Processing batches"):
            end = min(i + self.batch_size, len(X))
            out.append(process_func(X[i:end]))
            gc.collect()
        return np.vstack(out)

    # ------------------------------------------------------------------
    def fit(self, X, y):
        n_samples, H, W, C = X.shape
        print(f"Training PRP_PIL "
              f"[multi_scale={self.use_multi_scale}, "
              f"attn={self.use_attention}, "
              f"rp={self.use_random_projection}, "
              f"pca={self.use_pca}, "
              f"gabor={self.use_gabor}, "
              f"rff={self.use_rff}, "
              f"hash={self.use_binary_hashing}, "
              f"graph={self.use_graph_laplacian}, "
              f"stdz={self.use_feature_standardization}]")

        # --- Random projection matrices per scale ---
        self.R_list = []
        for ps, pd in zip(self.patch_sizes, self.projection_dims):
            D = ps * ps * C
            scale = 1.0 / np.sqrt(D)
            self.R_list.append(
                self.rng.normal(scale=scale, size=(D, pd))
                    .astype(np.float32))

        # --- Random Fourier Features ---
        if self.use_rff:
            D_img = H * W * C
            self.R_rff = self.rng.normal(
                0, np.sqrt(2 * self.rff_gamma),
                size=(D_img, self.rff_dim)).astype(np.float32)
            self.b_rff = self.rng.uniform(
                0, 2 * np.pi, size=(self.rff_dim,)).astype(np.float32)

        # --- Binary hashing matrices (one per distinct Z width) ---
        self.R_hash_dict = {}
        if self.use_binary_hashing:
            for ps, pd in zip(self.patch_sizes, self.projection_dims):
                d_in = pd if self.use_random_projection else ps * ps * C
                if d_in in self.R_hash_dict:
                    continue
                self.R_hash_dict[d_in] = self.rng.normal(
                    scale=1.0 / np.sqrt(d_in),
                    size=(d_in, self.hashing_dim)).astype(np.float32)

        # --- Gabor filters ---
        if self.use_gabor:
            self.gabor_filters = self._create_gabor_filters(6)

        # --- One-hot labels ---
        self.encoder = OneHotEncoder(sparse_output=False)
        y_onehot = self.encoder.fit_transform(y.reshape(-1, 1))
        self.n_classes = y_onehot.shape[1]

        # --- Feature extraction ---
        print("Extracting features...")
        H_feat = self._process_in_batches(X, self._create_features_batch)
        print(f"Raw feature dim: {H_feat.shape[1]}")

        # --- Optional pre-PCA standardization ---
        self._feat_mean = None
        self._feat_std = None
        if self.use_feature_standardization:
            self._feat_mean = H_feat.mean(axis=0, keepdims=True)
            self._feat_std = H_feat.std(axis=0, keepdims=True) + 1e-8
            H_feat = (H_feat - self._feat_mean) / self._feat_std
            print("Standardized features before PCA")

        # --- PCA toggle ---
        self._pca_fitted = False
        if self.use_pca and H_feat.shape[1] > 500:
            n_components = min(500, H_feat.shape[0] - 1)
            self.pca = PCA(n_components=n_components,
                           random_state=self.random_state)
            H_feat = self.pca.fit_transform(H_feat)
            self._pca_fitted = True
            print(f"Reduced to {H_feat.shape[1]} dims via PCA")

        # --- Hidden layer ---
        L = H_feat.shape[1]
        scale_w = 1.0 / np.sqrt(L)
        self.W = self.rng.normal(
            scale=scale_w, size=(L, self.n_hidden)).astype(np.float32)
        self.b = self.rng.normal(
            scale=0.1, size=(1, self.n_hidden)).astype(np.float32)

        print("Computing hidden activations...")
        H_pil_list = []
        for i in range(0, len(H_feat), self.batch_size):
            end = min(i + self.batch_size, len(H_feat))
            H_pil_list.append(
                self._apply_activation(H_feat[i:end] @ self.W + self.b))
            gc.collect()

        # Upcast to float64 before the solve for small-lambda stability.
        H_pil = np.vstack(H_pil_list).astype(np.float64)

        # --- Closed-form solve ---
        print("Solving for output weights...")
        HtH = H_pil.T @ H_pil
        I = np.eye(HtH.shape[0], dtype=np.float64)

        if self.use_graph_laplacian:
            print("Building graph-Laplacian regularizer...")
            n_samples_h = H_pil.shape[0]
            n_use = min(n_samples_h, 1500)
            idx = self.rng.choice(n_samples_h, size=n_use, replace=False)
            W_graph = kneighbors_graph(
                H_pil[idx], n_neighbors=8, mode='connectivity',
                include_self=False)
            L_g = np.array(W_graph.sum(axis=1)).flatten()
            D_g = np.diag(L_g)
            L_graph_small = D_g - W_graph.toarray()
            L_full = np.zeros((n_samples_h, n_samples_h), dtype=np.float64)
            L_full[np.ix_(idx, idx)] = L_graph_small.astype(np.float64)
            A = HtH + self.regularization * I + \
                self.graph_lambda * H_pil.T @ L_full @ H_pil
        else:
            A = HtH + self.regularization * I

        self.beta = np.linalg.solve(
            A, H_pil.T @ y_onehot.astype(np.float64))

        del H_feat, H_pil, y_onehot
        gc.collect()
        print("Training completed.")
        return self

    # ------------------------------------------------------------------
    def _transform(self, X):
        H_feat = self._process_in_batches(X, self._create_features_batch)
        if self._feat_mean is not None:
            H_feat = (H_feat - self._feat_mean) / self._feat_std
        if self._pca_fitted and self.pca is not None:
            H_feat = self.pca.transform(H_feat)
        return self._apply_activation(H_feat @ self.W + self.b)

    def predict_proba(self, X):
        H_pil = self._transform(X).astype(np.float64)
        scores = H_pil @ self.beta
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores)
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))