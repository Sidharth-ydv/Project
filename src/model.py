"""
src/model.py
------------
Module 3: Deep Learning CNN Model for Medical Image Anomaly Classification

Implements :class:`MedicalCNNModel` with:
  - Standard 4-block CNN classifier
  - Symmetric autoencoder for unsupervised anomaly scoring
  - Hybrid CNN + engineered-feature model
  - Grad-CAM heatmap generation
  - Full training loop with callbacks
  - Model persistence and batch inference
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy TensorFlow import — lets the module load even without GPU/TF installed
# ---------------------------------------------------------------------------
_TF_IMPORTED = False
tf = None
keras = None


def _import_tf() -> None:
    global tf, keras, _TF_IMPORTED
    if _TF_IMPORTED:
        return
    import tensorflow as _tf  # type: ignore

    _tf.get_logger().setLevel("ERROR")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    tf = _tf
    keras = _tf.keras
    _TF_IMPORTED = True
    logger.info("TensorFlow %s loaded.", _tf.__version__)


class MedicalCNNModel:
    """CNN model suite for medical image anomaly detection and classification.

    Wraps three Keras model variants:
        1. ``build_model`` — Standard 4-block supervised CNN classifier.
        2. ``build_autoencoder`` — Symmetric encoder-decoder for reconstruction-
           error-based anomaly scoring.
        3. ``build_hybrid_model`` — CNN feature extractor concatenated with
           handcrafted features, followed by a dense classification head.

    Args:
        input_shape: ``(H, W, C)`` spatial dimensions of input images.
        num_classes: Number of output classes.
        class_names: Display names for each class index.
        model_save_path: Default path for ``save_model`` / ``load_model``.
        dropout_dense1: Dropout rate for the first dense layer.
        dropout_dense2: Dropout rate for the second dense layer.

    Raises:
        ImportError: If TensorFlow is not installed when a model method is called.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (128, 128, 1),
        num_classes: int = 3,
        class_names: Optional[List[str]] = None,
        model_save_path: Union[str, Path] = "models/best_model.keras",
        dropout_dense1: float = 0.5,
        dropout_dense2: float = 0.3,
    ) -> None:
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.model_save_path = Path(model_save_path).resolve()
        self.dropout_dense1 = dropout_dense1
        self.dropout_dense2 = dropout_dense2
        self.model: Optional[Any] = None        # main classifier
        self.autoencoder: Optional[Any] = None  # autoencoder
        self.encoder: Optional[Any] = None      # encoder half
        self.history: Optional[Any] = None      # training history
        logger.info(
            "MedicalCNNModel configured: input=%s | classes=%d | labels=%s",
            input_shape,
            num_classes,
            self.class_names,
        )

    # ------------------------------------------------------------------
    # Model Builders
    # ------------------------------------------------------------------

    def build_model(
        self,
        input_shape: Optional[Tuple[int, int, int]] = None,
        num_classes: Optional[int] = None,
    ) -> Any:
        """Build a ResNet-style CNN classifier with residual skip connections.

        Architecture::

            Input
            → Stem: Conv2D(32, 3)
            → ResBlock(64, stride=2)
            → ResBlock(128, stride=2)
            → ResBlock(256, stride=2)
            → ResBlock(512, stride=2)
            → GlobalAvgPool
            → Dense(256, relu) → Dropout(0.5)
            → Dense(num_classes, softmax)

        Residual connections prevent vanishing gradients and dramatically
        improve accuracy on small datasets.

        Args:
            input_shape: Override instance ``self.input_shape``.
            num_classes: Override instance ``self.num_classes``.

        Returns:
            Compiled Keras :class:`~tensorflow.keras.Model`.
        """
        _import_tf()
        shape  = input_shape or self.input_shape
        n_cls  = num_classes  or self.num_classes

        inputs = keras.Input(shape=shape, name="image_input")

        # Stem — large-kernel first conv to capture broad structure
        x = keras.layers.Conv2D(32, 5, padding="same", use_bias=False, name="stem_conv")(inputs)
        x = keras.layers.BatchNormalization(name="stem_bn")(x)
        x = keras.layers.Activation("relu", name="stem_relu")(x)

        # Residual blocks
        x = self._res_block(x,  64, stride=2, name="res1")
        x = self._res_block(x, 128, stride=2, name="res2")
        x = self._res_block(x, 256, stride=2, name="res3")
        x = self._res_block(x, 512, stride=2, name="res4")   # <-- used for Grad-CAM

        x = keras.layers.GlobalAveragePooling2D(name="gap")(x)

        # Classifier head
        x = keras.layers.Dense(256, activation="relu", name="dense1")(x)
        x = keras.layers.Dropout(self.dropout_dense1, name="drop1")(x)
        outputs = keras.layers.Dense(n_cls, activation="softmax", name="predictions")(x)

        self.model = keras.Model(inputs, outputs, name="MedicalResNet")
        logger.info("ResNet-style CNN built | params=%s", f"{self.model.count_params():,}")
        return self.model

    def _res_block(self, x: Any, filters: int, stride: int = 1, name: str = "res") -> Any:
        """Single residual block: two Conv2D + BN + ReLU, with a projection shortcut."""
        shortcut = x

        # Main path
        x = keras.layers.Conv2D(filters, 3, strides=stride, padding="same",
                                use_bias=False, name=f"{name}_conv1")(x)
        x = keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = keras.layers.Activation("relu", name=f"{name}_relu1")(x)

        x = keras.layers.Conv2D(filters, 3, padding="same",
                                use_bias=False, name=f"{name}_conv2")(x)
        x = keras.layers.BatchNormalization(name=f"{name}_bn2")(x)

        # Projection shortcut when dims change
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = keras.layers.Conv2D(filters, 1, strides=stride, padding="same",
                                           use_bias=False, name=f"{name}_proj")(shortcut)
            shortcut = keras.layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

        x = keras.layers.Add(name=f"{name}_add")([x, shortcut])
        x = keras.layers.Activation("relu", name=f"{name}_relu2")(x)
        return x


    def build_autoencoder(
        self,
        input_shape: Optional[Tuple[int, int, int]] = None,
        latent_dim: int = 128,
    ) -> Tuple[Any, Any, Any]:
        """Build a symmetric convolutional autoencoder for anomaly scoring.

        Args:
            input_shape: Override instance ``self.input_shape``.
            latent_dim: Size of the Dense bottleneck latent vector.

        Returns:
            Tuple of ``(autoencoder, encoder, decoder)`` Keras models.
        """
        _import_tf()
        shape = input_shape or self.input_shape
        H, W, C = shape

        inputs = keras.Input(shape=shape, name="ae_input")

        # Encoder
        x = keras.layers.Conv2D(32, 3, activation="relu", padding="same", strides=2)(inputs)
        x = keras.layers.Conv2D(64, 3, activation="relu", padding="same", strides=2)(x)
        x = keras.layers.Conv2D(128, 3, activation="relu", padding="same", strides=2)(x)
        x = keras.layers.Flatten()(x)
        latent = keras.layers.Dense(latent_dim, activation="relu", name="latent_code")(x)

        # Determine pre-flatten spatial shape
        enc_h = H // 8
        enc_w = W // 8
        pre_flat_units = enc_h * enc_w * 128

        # Decoder
        x = keras.layers.Dense(pre_flat_units, activation="relu", name="decode_dense")(latent)
        x = keras.layers.Reshape((enc_h, enc_w, 128))(x)
        x = keras.layers.Conv2DTranspose(128, 3, activation="relu", padding="same", strides=2)(x)
        x = keras.layers.Conv2DTranspose(64, 3, activation="relu", padding="same", strides=2)(x)
        x = keras.layers.Conv2DTranspose(32, 3, activation="relu", padding="same", strides=2)(x)
        decoded = keras.layers.Conv2D(C, 3, activation="sigmoid", padding="same", name="reconstruction")(x)

        autoencoder = keras.Model(inputs, decoded, name="Autoencoder")
        encoder = keras.Model(inputs, latent, name="Encoder")

        # Decoder standalone (latent input)
        latent_input = keras.Input(shape=(latent_dim,), name="latent_input")
        dec_x = autoencoder.layers[-6](latent_input)  # decode_dense
        for layer in autoencoder.layers[-5:]:
            dec_x = layer(dec_x)
        decoder = keras.Model(latent_input, dec_x, name="Decoder")

        self.autoencoder = autoencoder
        self.encoder = encoder
        logger.info(
            "Autoencoder built | encoder params=%s | total params=%s",
            f"{encoder.count_params():,}",
            f"{autoencoder.count_params():,}",
        )
        return autoencoder, encoder, decoder

    def build_hybrid_model(
        self,
        n_engineered_features: int = 50,
        input_shape: Optional[Tuple[int, int, int]] = None,
        num_classes: Optional[int] = None,
    ) -> Any:
        """Build a hybrid CNN + engineered-feature fusion classifier.

        The CNN extracts spatial features from the image; these are concatenated
        with handcrafted feature vectors before the final dense head.

        Args:
            n_engineered_features: Dimensionality of the engineered feature vector.
            input_shape: Override instance ``self.input_shape``.
            num_classes: Override instance ``self.num_classes``.

        Returns:
            Keras :class:`~tensorflow.keras.Model` with two inputs.
        """
        _import_tf()
        shape = input_shape or self.input_shape
        n_cls = num_classes or self.num_classes

        # Image branch
        img_input = keras.Input(shape=shape, name="image")
        x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(img_input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(2)(x)
        x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(2)(x)
        x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(256, activation="relu")(x)
        cnn_features = keras.layers.Dropout(0.3)(x)

        # Engineered features branch
        feat_input = keras.Input(shape=(n_engineered_features,), name="engineered_features")
        feat_branch = keras.layers.Dense(128, activation="relu")(feat_input)
        feat_branch = keras.layers.BatchNormalization()(feat_branch)
        feat_branch = keras.layers.Dense(64, activation="relu")(feat_branch)

        # Fusion
        fused = keras.layers.Concatenate()([cnn_features, feat_branch])
        fused = keras.layers.Dense(256, activation="relu")(fused)
        fused = keras.layers.Dropout(0.4)(fused)
        fused = keras.layers.Dense(128, activation="relu")(fused)
        outputs = keras.layers.Dense(n_cls, activation="softmax", name="hybrid_pred")(fused)

        hybrid = keras.Model(inputs=[img_input, feat_input], outputs=outputs, name="HybridModel")
        logger.info("Hybrid model built | params=%s", f"{hybrid.count_params():,}")
        return hybrid

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile_model(self, learning_rate: float = 1e-4, model: Optional[Any] = None,
                      label_smoothing: float = 0.05) -> None:
        """Compile *model* with Adam, categorical cross-entropy + label smoothing, and key metrics.

        Args:
            learning_rate: Initial Adam optimiser learning rate.
            model: Keras model to compile.  Defaults to ``self.model``.
            label_smoothing: Label-smoothing epsilon (0 = off, 0.05–0.1 = mild).

        Raises:
            RuntimeError: If no model has been built yet.
        """
        _import_tf()
        m = model or self.model
        if m is None:
            raise RuntimeError("No model found.  Call build_model() first.")

        m.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
            metrics=[
                "accuracy",
                keras.metrics.AUC(name="auc", multi_label=False),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
            ],
        )
        logger.info("Model compiled | lr=%.6f | label_smoothing=%.2f", learning_rate, label_smoothing)

    def compile_autoencoder(self, learning_rate: float = 1e-4) -> None:
        """Compile the autoencoder with MSE reconstruction loss.

        Args:
            learning_rate: Adam learning rate.
        """
        _import_tf()
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not built.  Call build_autoencoder() first.")
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
        )
        logger.info("Autoencoder compiled.")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        learning_rate: float = 1e-4,
        patience: int = 10,
        model_save_path: Optional[Union[str, Path]] = None,
        class_weights: Optional[Dict[int, float]] = None,
        use_augmentation: bool = True,
    ) -> Any:
        """Train the CNN classifier with optional augmentation and class-weight balancing.

        Args:
            X_train: Float32 array of shape ``(N, H, W)`` or ``(N, H, W, C)``.
            y_train: Integer label array of shape ``(N,)`` or one-hot ``(N, C)``.
            epochs: Maximum training epochs.
            batch_size: Mini-batch size.
            validation_split: Fraction of training data used for validation.
            learning_rate: Initial learning rate.
            patience: EarlyStopping patience (epochs without improvement).
            model_save_path: Override save path for best-model checkpoint.
            class_weights: Optional dict mapping class index → weight.
                If ``None``, weights are computed automatically from label distribution.
            use_augmentation: If True, apply on-the-fly augmentation during training.

        Returns:
            Keras :class:`~tensorflow.keras.callbacks.History` object.

        Raises:
            RuntimeError: If :py:meth:`build_model` has not been called.
        """
        _import_tf()
        if self.model is None:
            raise RuntimeError("Model not built.  Call build_model() first.")

        # Ensure 4-D input
        X = self._ensure_4d(X_train)

        # Integer labels expected
        if y_train.ndim > 1:
            y_int = np.argmax(y_train, axis=1)
        else:
            y_int = y_train.astype(np.int32)

        # Auto-compute balanced class weights
        if class_weights is None:
            from sklearn.utils.class_weight import compute_class_weight  # type: ignore
            unique_classes = np.unique(y_int)
            cw = compute_class_weight("balanced", classes=unique_classes, y=y_int)
            class_weights = {int(c): float(w) for c, w in zip(unique_classes, cw)}
            logger.info("Auto class weights: %s", class_weights)

        y = keras.utils.to_categorical(y_int, num_classes=self.num_classes)

        self.compile_model(learning_rate=learning_rate)

        save_path = Path(model_save_path or self.model_save_path).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
                mode="max",
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.4,
                patience=max(3, patience // 3),
                min_lr=1e-7,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                str(save_path),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
                mode="max",
            ),
        ]

        logger.info(
            "Starting training | samples=%d | epochs=%d | batch=%d | save=%s",
            len(X), epochs, batch_size, save_path,
        )
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )
        logger.info("Training complete.")
        return self.history

    def train_autoencoder(
        self,
        X_train: np.ndarray,
        epochs: int = 30,
        batch_size: int = 32,
        validation_split: float = 0.15,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """Train the autoencoder on normal (or all) images.

        Args:
            X_train: Float32 image array ``(N, H, W)`` or ``(N, H, W, C)``.
            epochs: Max training epochs.
            batch_size: Mini-batch size.
            validation_split: Validation fraction.
            save_path: Optional path to save best autoencoder weights.

        Returns:
            Keras History object.
        """
        _import_tf()
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not built.")

        self.compile_autoencoder()
        X = self._ensure_4d(X_train)

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        ]
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                keras.callbacks.ModelCheckpoint(str(save_path), monitor="val_loss", save_best_only=True)
            )

        return self.autoencoder.fit(
            X, X,  # autoencoder is trained to reconstruct its input
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict class probabilities for a single image.

        Args:
            image: Raw float32 array ``(H, W)`` or ``(H, W, C)``.

        Returns:
            1-D float32 probability array of length ``num_classes``.

        Raises:
            RuntimeError: If no model is loaded.
        """
        _import_tf()
        if self.model is None:
            raise RuntimeError("No model loaded.")

        x = self._ensure_4d(image[np.newaxis])  # (1, H, W, C)
        probs: np.ndarray = self.model.predict(x, verbose=0)[0]
        return probs.astype(np.float32)

    def predict_batch(
        self,
        images: np.ndarray,
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch inference returning class indices and probabilities.

        Args:
            images: Float32 array of shape ``(N, H, W)`` or ``(N, H, W, C)``.
            batch_size: Inference batch size.

        Returns:
            Tuple of:
                - ``class_indices`` – int32 array ``(N,)``
                - ``probabilities`` – float32 array ``(N, num_classes)``
        """
        _import_tf()
        if self.model is None:
            raise RuntimeError("No model loaded.")

        X = self._ensure_4d(images)
        probs: np.ndarray = self.model.predict(X, batch_size=batch_size, verbose=0)
        class_indices = np.argmax(probs, axis=1).astype(np.int32)
        return class_indices, probs.astype(np.float32)

    def reconstruction_error(self, images: np.ndarray) -> np.ndarray:
        """Compute per-image MSE reconstruction error via the autoencoder.

        Args:
            images: Float32 array ``(N, H, W)`` or ``(N, H, W, C)``.

        Returns:
            1-D float32 reconstruction error array of shape ``(N,)``.
        """
        _import_tf()
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not built.")

        X = self._ensure_4d(images)
        reconstructed = self.autoencoder.predict(X, verbose=0)
        errors: np.ndarray = np.mean((X - reconstructed) ** 2, axis=(1, 2, 3))
        return errors.astype(np.float32)

    # ------------------------------------------------------------------
    # Grad-CAM
    # ------------------------------------------------------------------

    def get_gradcam(
        self,
        image: np.ndarray,
        layer_name: str = "conv4",
        class_index: Optional[int] = None,
    ) -> np.ndarray:
        """Generate a Grad-CAM heatmap for a single image.

        Args:
            image: Float32 image array ``(H, W)`` or ``(H, W, C)``.
            layer_name: Target convolutional layer name.
            class_index: Class to explain.  If ``None``, uses the predicted class.

        Returns:
            Float32 heatmap array in [0, 1], same spatial dimensions as *image*.
        """
        _import_tf()
        if self.model is None:
            raise RuntimeError("No model loaded.")

        # Build grad model
        try:
            conv_layer = self.model.get_layer(layer_name)
        except ValueError:
            logger.warning("Layer '%s' not found; using last conv layer.", layer_name)
            for layer in reversed(self.model.layers):
                if isinstance(layer, keras.layers.Conv2D):
                    conv_layer = layer
                    break
            else:
                return np.zeros(image.shape[:2], dtype=np.float32)

        grad_model = keras.Model(
            inputs=self.model.inputs,
            outputs=[conv_layer.output, self.model.output],
        )

        x = self._ensure_4d(image[np.newaxis])  # (1, H, W, C)

        with tf.GradientTape() as tape:
            inputs_tf = tf.cast(x, tf.float32)
            conv_outputs, predictions = grad_model(inputs_tf, training=False)
            if class_index is None:
                class_index = int(tf.argmax(predictions[0]))
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)          # (1, h, w, filters)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (filters,)

        conv_outputs_np = conv_outputs[0].numpy()          # (h, w, filters)
        pooled_grads_np = pooled_grads.numpy()             # (filters,)

        # Weight feature maps by pooled gradients
        heatmap = conv_outputs_np @ pooled_grads_np        # (h, w)
        heatmap = np.maximum(heatmap, 0)                   # ReLU

        # Normalise to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        else:
            heatmap = np.zeros_like(heatmap, dtype=np.float32)

        # Resize to input image spatial dims
        H, W = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)

        return heatmap_resized.astype(np.float32)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate model on held-out test data.

        Args:
            X_test: Float32 image array ``(N, H, W)`` or ``(N, H, W, C)``.
            y_test: Integer label array ``(N,)`` or one-hot ``(N, C)``.

        Returns:
            Dictionary with keys: ``loss``, ``accuracy``, ``auc``,
            ``precision``, ``recall``, ``classification_report``,
            ``confusion_matrix``, ``y_pred``, ``y_probs``.
        """
        from sklearn.metrics import classification_report, confusion_matrix  # type: ignore

        _import_tf()
        if self.model is None:
            raise RuntimeError("No model loaded.")

        X = self._ensure_4d(X_test)
        if y_test.ndim > 1:
            y_true = np.argmax(y_test, axis=1)
        else:
            y_true = y_test.astype(np.int32)

        y_true_cat = keras.utils.to_categorical(y_true, num_classes=self.num_classes)
        metrics = self.model.evaluate(X, y_true_cat, verbose=0)
        metric_names = ["loss", "accuracy", "auc", "precision", "recall"]

        _, y_probs = self.predict_batch(X_test)
        y_pred = np.argmax(y_probs, axis=1)

        report = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )
        cm = confusion_matrix(y_true, y_pred)

        result = {name: float(val) for name, val in zip(metric_names, metrics)}
        result["classification_report"] = report
        result["confusion_matrix"] = cm
        result["y_pred"] = y_pred
        result["y_probs"] = y_probs

        logger.info(
            "Evaluation | acc=%.3f | auc=%.3f | precision=%.3f | recall=%.3f",
            result["accuracy"],
            result["auc"],
            result["precision"],
            result["recall"],
        )
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save model weights and architecture to disk.

        Args:
            path: Optional override for save path.
        """
        _import_tf()
        if self.model is None:
            raise RuntimeError("No model to save.")
        p = Path(path or self.model_save_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(p))
        logger.info("CNN model saved → %s", p)

    def load_model(self, path: Optional[Union[str, Path]] = None) -> Any:
        """Load a previously saved Keras model.

        Args:
            path: Optional override for load path.

        Returns:
            Loaded Keras model (also stored in ``self.model``).
        """
        _import_tf()
        p = Path(path or self.model_save_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")
        self.model = keras.models.load_model(str(p))
        logger.info("CNN model loaded from %s", p)
        return self.model

    def save_autoencoder(self, path: Union[str, Path]) -> None:
        """Save the autoencoder to disk.

        Args:
            path: Save path.
        """
        _import_tf()
        if self.autoencoder is None:
            raise RuntimeError("No autoencoder to save.")
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        self.autoencoder.save(str(p))
        logger.info("Autoencoder saved → %s", p)

    def load_autoencoder(self, path: Union[str, Path]) -> Any:
        """Load a saved autoencoder from disk.

        Args:
            path: File path.

        Returns:
            Loaded Keras autoencoder model.
        """
        _import_tf()
        p = Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Autoencoder file not found: {p}")
        self.autoencoder = keras.models.load_model(str(p))
        logger.info("Autoencoder loaded from %s", p)
        return self.autoencoder

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_4d(self, x: np.ndarray) -> np.ndarray:
        """Ensure array is 4-D ``(N, H, W, C)`` suitable for Keras."""
        if x.ndim == 2:
            x = x[np.newaxis, ..., np.newaxis]
        elif x.ndim == 3:
            if x.shape[0] in (1, 3):  # likely (C, H, W) single image
                x = np.transpose(x, (1, 2, 0))[np.newaxis]
            else:
                # (N, H, W) batch
                x = x[..., np.newaxis]
        elif x.ndim == 4:
            pass
        else:
            raise ValueError(f"Unexpected input ndim={x.ndim}")
        return x.astype(np.float32)
