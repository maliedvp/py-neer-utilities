from pathlib import Path
import pickle
from neer_match.matching_model import DLMatchingModel, NSMatchingModel
from neer_match.similarity_map import SimilarityMap
import tensorflow as tf
from tensorflow.keras import layers as _layers
from graphviz import Digraph
from typing import Dict, List

import typing
import shutil

import numpy as np

from neer_match_utilities.thresholds import pair_labels, scores_at, tune_threshold

import pandas as pd


class Model:
    """
    A class for saving and loading matching models.

    Methods
    -------
    save(model, target_directory, name):
        Save the specified model to a target directory.
    load(model_directory):
        Load a model from a given directory.
    """

    @staticmethod
    def save(
        model: typing.Union["DLMatchingModel", "NSMatchingModel"],
        target_directory: Path,
        name: str,
        save_architecture: bool = False,
        overwrite: bool = False,
    ) -> None:
        """
        Save the model to a specified directory.

        Parameters
        ----------
        model : DLMatchingModel or NSMatchingModel
            The model to be saved.
        target_directory : Path
            The directory where the model should be saved.
        name : str
            Name of the model directory.
        save_architecture : bool, optional
            Whether to also render the architecture as ``architecture.png``.
            Requires the graphviz binaries. Defaults to False.
        overwrite : bool, optional
            Whether to replace a model already saved under ``name``. Defaults to
            False, in which case an existing model raises ``FileExistsError``.

        Raises
        ------
        FileExistsError
            If ``target_directory / name / "model"`` exists and ``overwrite`` is False.
        """

        target_directory = Path(target_directory) / name / "model"

        if target_directory.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Directory '{target_directory}' already exists. "
                    "Pass overwrite=True to replace the saved model."
                )
            shutil.rmtree(target_directory)

        target_directory.mkdir(parents=True, exist_ok=True)

        # --- Build composite similarity info ---
        # Use the original instructions stored in the SimilarityMap.
        # We assume model.similarity_map.instructions is a dict: { field: [metric1, metric2, ...], ... }
        instructions = model.similarity_map.instructions
        fields = list(instructions.keys())
        association_sizes = model.similarity_map.association_sizes()  # aggregated sizes per field
        composite_similarity_info = {}
        for i, field in enumerate(fields):
            agg_size = association_sizes[i]
            metrics = instructions[field]  # list of metric names as originally provided
            composite_similarity_info[field] = {
                "metrics": metrics,
                "aggregated_size": agg_size,
                "per_metric_size": agg_size // len(metrics)
            }

        # --- Save model initialization parameters from the record pair network ---
        model_params = {
            "initial_feature_width_scales": model.record_pair_network.initial_feature_width_scales,
            "feature_depths": model.record_pair_network.feature_depths,
            "initial_record_width_scale": model.record_pair_network.initial_record_width_scale,
            "record_depth": model.record_pair_network.record_depth,
        }

        # Save a composite dictionary containing both similarity info and model
        # parameters, and the classification threshold TrainingPipe tuned, if any
        composite_save = {
            "similarity_info": composite_similarity_info,
            "model_params": model_params,
            "best_threshold": getattr(model, "best_threshold_", None),
        }
        with open(target_directory / "model_info.pkl", "wb") as f:
            pickle.dump(composite_save, f)
        # --- End composite info saving ---

        if isinstance(model, DLMatchingModel):
            model.save_weights(target_directory / "model.weights.h5")
            if hasattr(model, "optimizer") and model.optimizer:
                optimizer_config = {
                    "class_name": model.optimizer.__class__.__name__,
                    "config": model.optimizer.get_config(),
                }
                with open(target_directory / "optimizer.pkl", "wb") as f:
                    pickle.dump(optimizer_config, f)
        elif isinstance(model, NSMatchingModel):
            model.record_pair_network.save_weights(target_directory / "record_pair_network.weights.h5")
            if hasattr(model, "optimizer") and model.optimizer:
                optimizer_config = {
                    "class_name": model.optimizer.__class__.__name__,
                    "config": model.optimizer.get_config(),
                }
                with open(target_directory / "optimizer.pkl", "wb") as f:
                    pickle.dump(optimizer_config, f)
        else:
            raise ValueError("The model must be an instance of DLMatchingModel or NSMatchingModel")
        
        # --- Optionally save architecture diagram ---
        if save_architecture:
            try:
                # Decide which Keras model to visualize in the standard plot
                if isinstance(model, DLMatchingModel):
                    base_model = model
                elif isinstance(model, NSMatchingModel):
                    base_model = model.record_pair_network
                else:
                    base_model = None

                # High-level field + record network diagram (DLMatchingModel only)
                if isinstance(model, DLMatchingModel):
                    field_sizes, record_sizes = _extract_two_stage_arch(model)
                    _draw_highlevel_architecture(
                        field_sizes,
                        record_sizes,
                        out_path=target_directory / "architecture.png",
                    )

            except Exception as e:
                print(f"Warning: could not save architecture diagram: {e}")

        print(f"Model successfully saved to {target_directory}")

    @staticmethod
    def load(model_directory: Path) -> typing.Union[DLMatchingModel, NSMatchingModel]:
        """
        Load a model from a specified directory.

        Parameters
        ----------
        model_directory : Path
            The directory containing the saved model.

        Returns
        -------
        DLMatchingModel or NSMatchingModel
            The loaded model. A DLMatchingModel carries the threshold tuned by
            TrainingPipe as ``best_threshold_`` (None if none was tuned) and,
            if there is one, is compiled with it.
        """

        model_directory = Path(model_directory) / "model"
        if not model_directory.exists():
            raise FileNotFoundError(f"Model directory '{model_directory}' does not exist.")

        # --- Load composite model info (similarity info and model parameters) ---
        with open(model_directory / "model_info.pkl", "rb") as f:
            composite_save = pickle.load(f)
        composite_similarity_info = composite_save["similarity_info"]
        model_params = composite_save["model_params"]

        # Reconstruct the original similarity_map as expected by DLMatchingModel:
        # (a plain dict mapping each field to its list of metric names)
        original_similarity_map = {field: info["metrics"] for field, info in composite_similarity_info.items()}

        # IMPORTANT: Reconstruct a SimilarityMap instance from the plain dict.
        similarity_map_instance = SimilarityMap(original_similarity_map)

        # Compute aggregated sizes in the order of fields.
        fields = list(composite_similarity_info.keys())
        aggregated_sizes = [composite_similarity_info[field]["aggregated_size"] for field in fields]
        # --- End loading composite info ---

        if (model_directory / "model.weights.h5").exists():
            # Initialize the model using the reconstructed SimilarityMap instance and stored parameters.
            model = DLMatchingModel(
                similarity_map=similarity_map_instance,
                initial_feature_width_scales=model_params["initial_feature_width_scales"],
                feature_depths=model_params["feature_depths"],
                initial_record_width_scale=model_params["initial_record_width_scale"],
                record_depth=model_params["record_depth"],
            )
            input_shapes = [tf.TensorShape([None, s]) for s in aggregated_sizes]
            model.build(input_shapes=input_shapes)

            # --- Build dummy inputs as a list of tensors (one per field) ---
            # Each dummy tensor has shape (1, aggregated_size) for that field.
            dummy_tensors = [
                tf.zeros((1, composite_similarity_info[field]["aggregated_size"]))
                for field in fields
            ]
            # --- End dummy inputs ---

            _ = model(dummy_tensors)  # Forward pass to instantiate all sublayers.
            model.load_weights(model_directory / "model.weights.h5")

            if (model_directory / "optimizer.pkl").exists():
                with open(model_directory / "optimizer.pkl", "rb") as f:
                    optimizer_config = pickle.load(f)
                optimizer_class = getattr(tf.keras.optimizers, optimizer_config["class_name"])
                model.optimizer = optimizer_class.from_config(optimizer_config["config"])

            # A model saved with a tuned threshold comes back compiled with it,
            # so that evaluate uses it. neer-match's compile sets the threshold
            # to 0.5 unless it is passed: compile it again with
            # threshold=model.best_threshold_.
            model.best_threshold_ = composite_save.get("best_threshold")
            if model.best_threshold_ is not None:
                model.compile(
                    optimizer=getattr(model, "optimizer", None),
                    threshold=model.best_threshold_,
                )
        elif (model_directory / "record_pair_network.weights.h5").exists():
            model = NSMatchingModel(similarity_map_instance)
            model.compile()
            model.record_pair_network.load_weights(model_directory / "record_pair_network.weights.h5")

            if (model_directory / "optimizer.pkl").exists():
                with open(model_directory / "optimizer.pkl", "rb") as f:
                    optimizer_config = pickle.load(f)
                optimizer_class = getattr(tf.keras.optimizers, optimizer_config["class_name"])
                model.optimizer = optimizer_class.from_config(optimizer_config["config"])
        else:
            raise ValueError("Invalid model directory: neither DLMatchingModel nor NSMatchingModel was detected.")

        return model


def _extract_two_stage_arch(dl_model: "DLMatchingModel"):
    """
    Extract architecture of field networks and record-pair network.

    Returns
    -------
    field_sizes : dict[str, list[int]]
        {field_name: [in_dim, h1, ..., out_dim]} for each field network.
        The key is a human-readable label like "current_name ~ alternative_name".
    record_sizes : list[int]
        [in_dim, h1, ..., out_dim] for the record-pair network.
    """
    # Work on inner RecordPairNetwork if present
    base = getattr(dl_model, "record_pair_network", dl_model)

    field_sizes: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------
    # Field networks: list of FieldPairNetwork + association keys
    # ------------------------------------------------------------------
    nets = getattr(base, "field_networks", None)
    sim_map = getattr(base, "similarity_map", None)

    if nets is not None and sim_map is not None:
        # Association keys, e.g. "current_name~alternative_name", "lat~lat_noise"
        assoc_keys = list(sim_map.instructions.keys())

        # Zip in order: one field net per association
        for key, net in zip(assoc_keys, nets):
            # Build a nice label using "~" as in your original similarity map
            parts = [p.strip() for p in key.split("~")]
            if len(parts) == 1:
                label = parts[0]
            else:
                label = f"{parts[0]} ~ {parts[1]}"

            sizes: List[int] = []

            # Input dimension: FieldPairNetwork.size if available
            in_dim = getattr(net, "size", None)
            if in_dim is not None:
                sizes.append(int(in_dim))

            # Hidden + output units: from net.field_layers if present,
            # otherwise from the Keras layers directly
            field_layers = getattr(net, "field_layers", None)
            if field_layers is None:
                field_layers = [
                    lyr for lyr in getattr(net, "layers", [])
                    if isinstance(lyr, _layers.Dense)
                ]

            for lyr in field_layers:
                if isinstance(lyr, _layers.Dense):
                    sizes.append(int(lyr.units))

            if sizes:
                field_sizes[label] = sizes

    # ------------------------------------------------------------------
    # Fallback: approximate from similarity_map + hyperparams
    # ------------------------------------------------------------------
    if not field_sizes and sim_map is not None:
        assoc_keys = list(sim_map.instructions.keys())
        assoc_sizes = sim_map.association_sizes()

        init_scale = getattr(base, "initial_feature_width_scales", None)
        depth = getattr(base, "feature_depths", None)

        for key, in_dim in zip(assoc_keys, assoc_sizes):
            parts = [p.strip() for p in key.split("~")]
            if len(parts) == 1:
                label = parts[0]
            else:
                label = f"{parts[0]} ~ {parts[1]}"

            sizes = [int(in_dim)]
            if init_scale is not None and depth is not None:
                width = int(in_dim * init_scale)
                for _ in range(int(depth)):
                    sizes.append(width)
            sizes.append(1)  # scalar field prediction
            field_sizes[label] = sizes

    # ------------------------------------------------------------------
    # Record-pair (record) network
    # ------------------------------------------------------------------
    record_sizes: List[int] = []

    rec_layers = getattr(base, "record_layers", None)
    if rec_layers:
        first_dense = next(
            (lyr for lyr in rec_layers if isinstance(lyr, _layers.Dense)),
            None,
        )
        if first_dense is not None:
            in_dim = int(first_dense.kernel.shape[0])
            record_sizes.append(in_dim)

        for lyr in rec_layers:
            if isinstance(lyr, _layers.Dense):
                record_sizes.append(int(lyr.units))

    return field_sizes, record_sizes


def _draw_highlevel_architecture(field_sizes: Dict[str, List[int]],
                                 record_sizes: List[int],
                                 out_path: Path):
    """
    Draw a high-level two-stage architecture diagram using graphviz.
    """

    g = Digraph("DLMatchingModel", format="png")
    g.attr(
        rankdir="TB",
        fontsize="12",
        labelloc="t",
        label="DLMatchingModel",
        fontname="Helvetica-Bold",
    )

    # ----------------------------
    # Field Networks cluster
    # ----------------------------
    with g.subgraph(name="cluster_fields") as c:
        c.attr(
            label="Field Networks",
            style="rounded",
            color="black",
            labelloc="t",
            fontsize="12",
            fontname="Helvetica",
        )

        field_pred_nodes = []

        for i, (field, sizes) in enumerate(field_sizes.items()):
            # Field network node (pink)
            net_label = f"Field Network\n({field})\\n" + " → ".join(str(s) for s in sizes)
            net_node = f"field_{i}_net"
            c.node(
                net_node,
                label=net_label,
                shape="box",
                style="rounded,filled",
                fillcolor="#f4cccc",      # light pink
                fontname="Helvetica",
                fontsize="10",
            )

            # Field prediction node (yellow)
            pred_label = f"Field\nPrediction"
            pred_node = f"field_{i}_pred"
            c.node(
                pred_node,
                label=pred_label,
                shape="box",
                style="rounded,filled",
                fillcolor="#fff2cc",      # light yellow
                fontname="Helvetica",
                fontsize="10",
            )

            # Edge: field network -> field prediction
            c.edge(net_node, pred_node)

            field_pred_nodes.append(pred_node)

    # ----------------------------
    # Record Network
    # ----------------------------
    if record_sizes:
        rec_label = "Record Network\\n" + " → ".join(str(s) for s in record_sizes)
    else:
        rec_label = "Record Network"

    g.node(
        "record_net",
        rec_label,
        shape="box",
        style="rounded,filled",
        fillcolor="#f4cccc",
        fontname="Helvetica",
        fontsize="11",
    )

    # ----------------------------
    # Record Prediction
    # ----------------------------
    g.node(
        "record_pred",
        "Record\nPrediction",
        shape="box",
        style="rounded,filled",
        fillcolor="#fff2cc",
        fontname="Helvetica",
        fontsize="11",
    )

    # ----------------------------
    # Arrows from all field predictions to record net
    # ----------------------------
    for pn in field_pred_nodes:
        g.edge(pn, "record_net")

    g.edge("record_net", "record_pred")

    # ----------------------------
    # Render
    # ----------------------------
    out_stem = out_path.with_suffix("")  # Graphviz appends .png
    g.render(filename=str(out_stem), format="png", cleanup=True)


class EpochEndSaver(tf.keras.callbacks.Callback):
    """
    Custom Keras callback to save weights and biases at the end of every epoch
    using the `Model.save(...)` static method.
    """

    def __init__(self, base_dir: Path, model_name: str):
        """
        Parameters
        ----------
        base_dir : Path
            The root directory under which the model subdirectories will be created.
            For instance: Path.cwd(), as used by TrainingPipe
        model_name : str
            A short identifier for the model. Each epoch’s directory will be
            base_dir / model_name / "checkpoints" / "epoch_<NN>"
        """
        super().__init__()
        self.base_dir = Path(base_dir)
        self.model_name = model_name

    def on_epoch_end(self, epoch: int, logs=None):
        """
        At the end of each epoch, call Model.save(...) so that
        weights and optimizer state are pickled as per your spec.
        """
        # epoch is zero‐indexed, but we probably want to save as "epoch_01", etc.
        epoch_index = epoch + 1
        epoch_dir_name = f"epoch_{epoch_index:02d}"
        
        # Build the directory where we want to dump model info & weights
        target_directory = self.base_dir / self.model_name / "checkpoints"


        # Ensure the parent directory exists; your Model.save(...) will
        # create the exact "model" subdirectory under this path.
        target_directory.mkdir(parents=True, exist_ok=True)

        # The checkpoints `self.model` attribute is the actual keras.Model (or subclass).
        ## We only proceed if it’s an instance of DLMatchingModel or NSMatchingModel:
        if not isinstance(self.model, (DLMatchingModel, NSMatchingModel)):
            raise ValueError(
                f"`EpochEndSaver` expected DLMatchingModel or NSMatchingModel, got {type(self.model)}"
            )

        # Now call your custom save function. A checkpoint left behind by an
        # interrupted run is stale, so it is replaced rather than kept.
        Model.save(
            model=self.model,
            target_directory=target_directory,
            name=epoch_dir_name,
            overwrite=True,
        )


def validation_scores(
    model: DLMatchingModel,
    left: pd.DataFrame,
    right: pd.DataFrame,
    matches: pd.DataFrame,
    batch_size: int = 1024,
    tune_metric: str | None = None,
) -> tuple[dict, np.ndarray]:
    """
    Score a model on all record pairs of validation data.

    Returns the scores, keyed as an epoch's logs are, and the predicted
    probabilities of all pairs, left record by left record:

    - ``val_TP`` ... ``val_MCC``: at the model's threshold, 0.5 unless it was
      compiled with another. Precision, Recall, F1 and MCC are 0 where they are
      undefined.
    - ``val_Loss``: the model's loss on all pairs at once.
    - With ``tune_metric`` ('mcc' or 'f1'): ``val_threshold``, the threshold
      that maximises it (see :func:`~neer_match_utilities.thresholds.tune_threshold`),
      and ``val_tuned_F1``, ``val_tuned_MCC``, ``val_tuned_Precision`` and
      ``val_tuned_Recall`` at that threshold.
    """
    probabilities = np.asarray(
        model.predict(left, right, batch_size=batch_size, verbose=0), dtype=np.float32
    ).ravel()
    labels = pair_labels(matches, len(left), len(right))

    threshold = float(getattr(model, "_threshold", 0.5))
    scores = {f"val_{name}": value for name, value in scores_at(labels, probabilities, threshold).items()}

    loss = getattr(model, "loss", None)
    if callable(loss):
        value = loss(
            tf.constant(labels, dtype=tf.float32), tf.constant(probabilities, dtype=tf.float32)
        )
        scores["val_Loss"] = float(tf.reduce_mean(value))

    if tune_metric is not None:
        tuned_threshold, tuned = tune_threshold(labels, probabilities, tune_metric)
        scores["val_threshold"] = tuned_threshold
        for name in ("F1", "MCC", "Precision", "Recall"):
            scores[f"val_tuned_{name}"] = tuned[name]

    return scores, probabilities


class ValidationMetrics(tf.keras.callbacks.Callback):
    """
    Score the model on validation data at the end of every epoch.

    The scores of :func:`validation_scores` on all validation record pairs are
    added to the epoch's logs (``val_F1``, ``val_MCC``, ``val_Loss``, ...;
    with ``tune_metric`` also ``val_threshold`` and ``val_tuned_MCC``, ...),
    where the history and later callbacks, such as EarlyStopping, see them.

    It predicts from data frames, which every supported neer-match version
    provides; Keras's own ``validation_data`` needs a newer neer-match.
    """

    def __init__(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        matches: pd.DataFrame,
        batch_size: int = 1024,
        tune_metric: str | None = None,
    ):
        """
        Parameters
        ----------
        left, right, matches : pd.DataFrame
            The validation data, in the format the model is trained on.
        batch_size : int
            Record pairs per prediction batch. Affects speed and memory only.
        tune_metric : {'mcc', 'f1'} or None
            If given, each epoch is also scored at the threshold that
            maximises this metric on the validation pairs.
        """
        super().__init__()
        self.left, self.right, self.matches = left, right, matches
        self.batch_size = batch_size
        self.tune_metric = tune_metric

    def on_epoch_end(self, epoch: int, logs=None):
        scores, _ = validation_scores(
            self.model, self.left, self.right, self.matches,
            batch_size=self.batch_size, tune_metric=self.tune_metric,
        )
        if logs is not None:
            logs.update(scores)
