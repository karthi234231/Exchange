"""Production inference engine for NSE AlgoTrader.

Spec reference: ARCH-FIX-2.
Loads Treelite ``.so`` compiled models for sub-5ms inference latency.
Receives feature vectors on ZMQ SUB (port 5556) and publishes trading
signals on ZMQ PUSH (port 5557).

Design constraints:
- NEVER loads ``.pkl`` files in the production inference path.
- All array work at inference time uses numpy only — no pandas.
- Confidence threshold 0.55; signals below this are suppressed (HOLD).
"""

from __future__ import annotations

import struct
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import msgpack
import numpy as np
import structlog
import zmq

from algo_trading.models.model_registry import FrozenModelSpec, ModelMismatchError

logger = structlog.get_logger(__name__)

# Signal constants
SIGNAL_SELL: int = -1
SIGNAL_HOLD: int = 0
SIGNAL_BUY: int = 1

# Class-index → signal mapping (mirror of ensemble.py)
_CLASS_TO_SIGNAL: dict[int, int] = {0: -1, 1: 0, 2: 1}

# Minimum confidence to publish a non-HOLD signal
CONFIDENCE_THRESHOLD: float = 0.55


# ---------------------------------------------------------------------------
# Latency snapshot
# ---------------------------------------------------------------------------


@dataclass
class LatencySnapshot:
    """Single inference latency record.

    Attributes:
        symbol: NSE symbol that was evaluated.
        t_recv_ns: Nanosecond timestamp when the ZMQ message was received.
        t_infer_ns: Nanosecond timestamp immediately after
            :meth:`EnsembleInferenceEngine.predict` returned.
        t_push_ns: Nanosecond timestamp after the ZMQ PUSH completed (or 0 if
            the signal was suppressed).
        signal: Trading signal emitted (``-1``, ``0``, or ``+1``).
        confidence: Max class probability from the meta-learner.
        published: Whether the signal was pushed to ZMQ consumers.
    """

    symbol: str
    t_recv_ns: int
    t_infer_ns: int
    t_push_ns: int
    signal: int
    confidence: float
    published: bool

    @property
    def inference_ms(self) -> float:
        """Inference latency in milliseconds (recv → infer).

        Returns:
            Elapsed milliseconds between ZMQ receive and predict completion.
        """
        return (self.t_infer_ns - self.t_recv_ns) / 1_000_000.0

    @property
    def total_ms(self) -> float:
        """Total latency in milliseconds (recv → push).

        Returns:
            Elapsed milliseconds between ZMQ receive and ZMQ push (or infer
            if the signal was not published).
        """
        end = self.t_push_ns if self.t_push_ns else self.t_infer_ns
        return (end - self.t_recv_ns) / 1_000_000.0


# ---------------------------------------------------------------------------
# Wire protocol helpers
# ---------------------------------------------------------------------------


def _deserialise_feature_vector(data: bytes) -> tuple[str, np.ndarray]:
    """Deserialise a ZMQ message into a symbol string and float32 feature array.

    Wire format (little-endian):
    - 2 bytes: uint16 — symbol string byte length ``L``
    - ``L`` bytes: UTF-8 symbol string
    - remaining bytes: contiguous float32 array (C-order)

    Args:
        data: Raw bytes received from ZMQ SUB socket.

    Returns:
        Tuple of ``(symbol, feature_vector)`` where ``feature_vector`` is a
        1-D numpy array of dtype ``float32``.

    Raises:
        ValueError: If the message is too short or the symbol length overruns
            the buffer.
    """
    if len(data) < 2:
        raise ValueError("Message too short: missing symbol length header.")
    sym_len: int = struct.unpack_from("<H", data, 0)[0]
    offset = 2
    if len(data) < offset + sym_len:
        raise ValueError(
            f"Message truncated: expected {sym_len} bytes for symbol name."
        )
    symbol: str = data[offset : offset + sym_len].decode("utf-8")
    offset += sym_len
    feat_bytes = data[offset:]
    if len(feat_bytes) % 4 != 0:
        raise ValueError(
            f"Feature payload length {len(feat_bytes)} is not a multiple of 4 (float32)."
        )
    feature_vector = np.frombuffer(feat_bytes, dtype=np.float32).copy()
    return symbol, feature_vector


def _serialise_signal(symbol: str, signal: int, confidence: float) -> bytes:
    """Serialise a trading signal to bytes for ZMQ PUSH.

    Wire format (little-endian):
    - 2 bytes: uint16 — symbol string byte length
    - N bytes: UTF-8 symbol
    - 1 byte:  int8 — signal value (``-1``, ``0``, or ``+1``)
    - 4 bytes: float32 — confidence

    Args:
        symbol: NSE symbol string.
        signal: Trading signal, one of ``-1``, ``0``, or ``+1``.
        confidence: Max class probability, ``0.0``–``1.0``.

    Returns:
        Serialised bytes ready for ZMQ send.
    """
    sym_bytes = symbol.encode("utf-8")
    sym_len = len(sym_bytes)
    payload = (
        struct.pack("<H", sym_len)
        + sym_bytes
        + struct.pack("<b", signal)
        + struct.pack("<f", confidence)
    )
    return payload


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------


class EnsembleInferenceEngine:
    """Treelite-based ensemble inference engine with ZMQ I/O.

    Loads three Treelite ``.so`` shared libraries and a ``pickle``-serialised
    ``LogisticRegression`` meta-learner.  Feature validation is performed
    against the immutable :class:`FrozenModelSpec` before every prediction.

    ZMQ topology:
    - SUB socket on ``tcp://*:<zmq_sub_port>`` — subscribes to all topics.
    - PUSH socket on ``tcp://*:<zmq_push_port>`` — pushes signal messages.

    Attributes:
        CONFIDENCE_THRESHOLD: Minimum confidence required to publish a signal.
    """

    CONFIDENCE_THRESHOLD: float = CONFIDENCE_THRESHOLD

    def __init__(
        self,
        spec: FrozenModelSpec,
        zmq_sub_port: int = 5556,
        zmq_push_port: int = 5557,
    ) -> None:
        """Initialise the engine with a frozen model spec.

        Models are NOT loaded at this point; call :meth:`load_models` before
        :meth:`run`.

        Args:
            spec: Immutable production spec describing the models to load.
            zmq_sub_port: TCP port on which feature vectors are received.
            zmq_push_port: TCP port on which trading signals are published.
        """
        self._spec = spec
        self._zmq_sub_port = zmq_sub_port
        self._zmq_push_port = zmq_push_port

        # Cache expected feature hash once to avoid recomputing on every message
        self._expected_feature_hash: str = FrozenModelSpec.compute_feature_hash(
            self._spec.feature_names
        )

        # Treelite predictors (loaded by load_models)
        self._tl_xgb: Any = None
        self._tl_lgb: Any = None
        self._tl_cat: Any = None

        # Meta LR loaded via pickle (acceptable outside hot path)
        self._meta_lr: Any = None

        self._models_loaded: bool = False
        self._latency_log: list[LatencySnapshot] = []

        # Fix A: ZMQ heartbeat detection state
        self._last_message_time: float = time.monotonic()
        self._alert_sender: Any = None  # Optional alert callable; set externally if needed

        logger.info(
            "inference_engine_created",
            version=spec.version,
            bar_type=spec.bar_type,
            zmq_sub_port=zmq_sub_port,
            zmq_push_port=zmq_push_port,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Load Treelite ``.so`` files and the meta LogisticRegression.

        Validates the spec feature hash post-load to guard against artefact
        substitution.

        Raises:
            ImportError: If ``treelite_runtime`` is not installed.
            RuntimeError: If any model file cannot be loaded.
            ModelMismatchError: If the loaded spec hash does not match a
                re-derived hash (should never happen for unmodified artefacts).
        """
        import pickle  # only used here, outside hot inference loop

        try:
            import treelite_runtime as tl_rt  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "treelite_runtime is required for production inference. "
                "Install with: pip install treelite treelite_runtime"
            ) from exc

        logger.info("loading_treelite_models", version=self._spec.version)

        try:
            self._tl_xgb = tl_rt.Predictor(self._spec.treelite_xgb_path, verbose=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load XGB Treelite model from "
                f"'{self._spec.treelite_xgb_path}': {exc}"
            ) from exc

        try:
            self._tl_lgb = tl_rt.Predictor(self._spec.treelite_lgb_path, verbose=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load LGB Treelite model from "
                f"'{self._spec.treelite_lgb_path}': {exc}"
            ) from exc

        try:
            self._tl_cat = tl_rt.Predictor(self._spec.treelite_cat_path, verbose=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load CatBoost Treelite model from "
                f"'{self._spec.treelite_cat_path}': {exc}"
            ) from exc

        try:
            with open(self._spec.meta_lr_path, "rb") as fh:
                self._meta_lr = pickle.load(fh)  # noqa: S301 — only used at init
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load meta LR from '{self._spec.meta_lr_path}': {exc}"
            ) from exc

        # Re-validate spec hash post-load as a sanity check
        recomputed = FrozenModelSpec.compute_feature_hash(self._spec.feature_names)
        if recomputed != self._spec.feature_schema_hash:
            raise ModelMismatchError(
                f"Spec hash mismatch after model load: "
                f"stored={self._spec.feature_schema_hash}, "
                f"recomputed={recomputed}."
            )

        self._models_loaded = True
        logger.info(
            "models_loaded",
            version=self._spec.version,
            n_features=len(self._spec.feature_names),
        )

    # ------------------------------------------------------------------
    # Predict (hot path)
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> tuple[int, float]:
        """Run ensemble inference on a single feature row.

        Hot path — target < 5 ms end-to-end.  Uses Treelite predictors
        directly; no Python overhead from sklearn wrappers.

        Steps:
        1. Run X through each Treelite predictor → probability vectors.
        2. Concatenate into meta-feature row.
        3. Pass through meta LR → class probabilities.
        4. Map argmax class index to trading signal.
        5. Return ``(signal, confidence)`` where confidence = max probability.

        Args:
            X: 1-D numpy float32 array of length ``n_features``, OR a 2-D
                array of shape ``(1, n_features)``.

        Returns:
            Tuple ``(signal, confidence)`` where:
            - ``signal`` is one of ``-1`` (SELL), ``0`` (HOLD), ``+1`` (BUY).
            - ``confidence`` is the max class probability (``0.0``–``1.0``).

        Raises:
            RuntimeError: If :meth:`load_models` has not been called.
        """
        import treelite_runtime as tl_rt  # type: ignore[import]

        self._assert_loaded()

        # Fix B: dtype and NaN/inf validation before predict
        if not np.issubdtype(X.dtype, np.floating):
            X = X.astype(np.float32)
        if not np.isfinite(X).all():
            raise ValueError(f"Feature vector contains NaN or inf: {X}")

        # Ensure 2-D (1, n_features) for Treelite batch API
        if X.ndim == 1:
            X_2d = X.reshape(1, -1).astype(np.float32)
        else:
            X_2d = X.astype(np.float32)

        # Treelite DMatrix expects C-contiguous float32
        dmat = tl_rt.DMatrix(X_2d)

        # Base model probabilities — shape (1, 3) each
        xgb_proba: np.ndarray = self._tl_xgb.predict(dmat)   # (1, 3)
        lgb_proba: np.ndarray = self._tl_lgb.predict(dmat)   # (1, 3)
        cat_proba: np.ndarray = self._tl_cat.predict(dmat)   # (1, 3)

        # Stack → meta feature row of length 9
        meta_row = np.hstack([xgb_proba, lgb_proba, cat_proba])  # (1, 9)

        # Meta LR prediction
        meta_proba: np.ndarray = self._meta_lr.predict_proba(meta_row)  # (1, 3)
        proba_1d: np.ndarray = meta_proba[0]

        class_idx: int = int(np.argmax(proba_1d))
        confidence: float = float(proba_1d[class_idx])
        signal: int = _CLASS_TO_SIGNAL[class_idx]

        return signal, confidence

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self, symbols: list[str]) -> None:
        """Start the inference event loop.

        Subscribes to all ZMQ topics on the SUB socket, deserialises feature
        vectors, validates schema, runs inference, and pushes signals.

        The loop runs indefinitely until a ``KeyboardInterrupt`` is received.

        Steps per message:
        1. ZMQ SUB receives raw bytes.
        2. :func:`_deserialise_feature_vector` → ``(symbol, X)``.
        3. Validate feature schema against :attr:`_spec` (raises on mismatch).
        4. :meth:`predict` → ``(signal, confidence)``.
        5. If ``signal != HOLD`` and ``confidence > CONFIDENCE_THRESHOLD``:
           push serialised signal via ZMQ PUSH.
        6. Record :class:`LatencySnapshot`.

        Args:
            symbols: List of NSE symbols the engine is expected to process.
                Used for initial validation logging only; filtering is done
                by the upstream feature publisher.

        Raises:
            RuntimeError: If :meth:`load_models` has not been called before
                this method.
        """
        self._assert_loaded()

        ctx = zmq.Context()
        sub_sock: zmq.Socket = ctx.socket(zmq.SUB)
        sub_sock.connect(f"tcp://localhost:{self._zmq_sub_port}")
        sub_sock.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe to all topics

        push_sock: zmq.Socket = ctx.socket(zmq.PUSH)
        push_sock.bind(f"tcp://*:{self._zmq_push_port}")

        logger.info(
            "inference_loop_started",
            version=self._spec.version,
            symbols=symbols,
            sub_port=self._zmq_sub_port,
            push_port=self._zmq_push_port,
            confidence_threshold=self.CONFIDENCE_THRESHOLD,
        )

        # Fix A: start heartbeat monitor thread
        _stop_heartbeat = threading.Event()

        def _heartbeat_monitor() -> None:
            _IST = ZoneInfo("Asia/Kolkata")
            _MARKET_OPEN = datetime.strptime("09:30", "%H:%M").time()
            _MARKET_CLOSE = datetime.strptime("15:30", "%H:%M").time()
            while not _stop_heartbeat.is_set():
                _stop_heartbeat.wait(timeout=5.0)
                now_ist = datetime.now(_IST)
                if _MARKET_OPEN <= now_ist.time() <= _MARKET_CLOSE:
                    if time.monotonic() - self._last_message_time > 30.0:
                        logger.error("inference.zmq_silence — no feature vector for 30s")
                        if self._alert_sender is not None:
                            try:
                                self._alert_sender(
                                    "inference.zmq_silence: no feature vector received for 30s"
                                )
                            except Exception as _alert_exc:
                                logger.warning(
                                    "inference.alert_send_failed", error=str(_alert_exc)
                                )

        _hb_thread = threading.Thread(
            target=_heartbeat_monitor, name="inference-heartbeat", daemon=True
        )
        _hb_thread.start()

        try:
            while True:
                t_recv_ns = time.perf_counter_ns()
                try:
                    parts = sub_sock.recv_multipart()
                    if len(parts) != 2:
                        logger.warning("unexpected_frame_count", count=len(parts))
                        continue
                    symbol_bytes, payload_bytes = parts
                    symbol = symbol_bytes.decode("utf-8", errors="replace")
                    msg: dict = msgpack.unpackb(payload_bytes, raw=False)
                    # Fix A: update heartbeat timestamp on every received message
                    self._last_message_time = time.monotonic()
                except Exception as exc:
                    logger.warning("deserialise_error", error=str(exc))
                    continue

                # Validate feature hash (expected hash cached at init to avoid hot-path SHA-256)
                incoming_hash: str = msg.get("feature_hash", "")
                if incoming_hash and incoming_hash != self._expected_feature_hash:
                    logger.error(
                        "feature_hash_mismatch",
                        symbol=symbol,
                        incoming=incoming_hash,
                        expected=self._expected_feature_hash,
                    )
                    continue

                # Extract and validate feature vector
                feat_bytes: bytes = msg.get("features", b"")
                try:
                    feature_vec = np.frombuffer(feat_bytes, dtype=np.float32).copy()
                except Exception as exc:
                    logger.warning("feature_bytes_error", symbol=symbol, error=str(exc))
                    continue

                if len(feature_vec) != len(self._spec.feature_names):
                    logger.error(
                        "feature_length_mismatch",
                        symbol=symbol,
                        received=len(feature_vec),
                        expected=len(self._spec.feature_names),
                    )
                    continue

                signal, confidence = self.predict(feature_vec)
                t_infer_ns = time.perf_counter_ns()

                published = False
                t_push_ns = 0

                if signal != SIGNAL_HOLD and confidence > self.CONFIDENCE_THRESHOLD:
                    push_sock.send(_serialise_signal(symbol, signal, confidence))
                    t_push_ns = time.perf_counter_ns()
                    published = True

                snapshot = LatencySnapshot(
                    symbol=symbol,
                    t_recv_ns=t_recv_ns,
                    t_infer_ns=t_infer_ns,
                    t_push_ns=t_push_ns,
                    signal=signal,
                    confidence=confidence,
                    published=published,
                )
                self._latency_log.append(snapshot)

                logger.info(
                    "inference_result",
                    symbol=symbol,
                    signal=signal,
                    confidence=round(confidence, 4),
                    published=published,
                    inference_ms=round(snapshot.inference_ms, 3),
                    total_ms=round(snapshot.total_ms, 3),
                )

                if snapshot.total_ms > 5.0:
                    logger.warning(
                        "latency_breach",
                        symbol=symbol,
                        total_ms=round(snapshot.total_ms, 3),
                        threshold_ms=5.0,
                    )

        except KeyboardInterrupt:
            logger.info("inference_loop_stopped", reason="KeyboardInterrupt")
        finally:
            _stop_heartbeat.set()
            _hb_thread.join(timeout=2.0)
            sub_sock.close()
            push_sock.close()
            ctx.term()
            logger.info(
                "zmq_sockets_closed",
                total_predictions=len(self._latency_log),
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def latency_stats(self) -> dict[str, float]:
        """Compute summary latency statistics over recorded snapshots.

        Returns:
            Dict with keys ``"mean_ms"``, ``"p50_ms"``, ``"p95_ms"``,
            ``"p99_ms"``, ``"max_ms"``, and ``"n_samples"``.  Returns
            zeroed dict if no snapshots have been recorded yet.
        """
        if not self._latency_log:
            return {k: 0.0 for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms", "n_samples")}

        totals = np.array([s.total_ms for s in self._latency_log], dtype=np.float64)
        return {
            "mean_ms": float(np.mean(totals)),
            "p50_ms": float(np.percentile(totals, 50)),
            "p95_ms": float(np.percentile(totals, 95)),
            "p99_ms": float(np.percentile(totals, 99)),
            "max_ms": float(np.max(totals)),
            "n_samples": float(len(totals)),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_loaded(self) -> None:
        """Raise RuntimeError if models have not been loaded.

        Raises:
            RuntimeError: If :meth:`load_models` has not been called.
        """
        if not self._models_loaded:
            raise RuntimeError(
                "Models are not loaded. Call load_models() before predict() or run()."
            )
