import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Callable, Tuple, Optional, Union, Any
import os
import gc
import torch
import logging
import torch.nn.functional as F

from experiments.intervention_experiment import *
from causal.causal_model import CausalModel
from neural.LM_units import *
from neural.model_units import *
from neural.featurizers import *
from neural.pipeline import LMPipeline

from experiments.pyvene_core import _prepare_intervenable_inputs

# Set up logging
logger = logging.getLogger(__name__)

def LM_loss_and_metric_fn(pipeline, intervenable_model, batch, model_units_list):
    """
    Calculate loss and evaluation metrics for language model interventions.
    
    This function evaluates intervention effects by:
    
    1. Preparing intervenable inputs from the batch
    2. Concatenating ground truth label tokens to the base inputs
       (e.g., if input has length 10 and labels length 3, creates sequence of length 13)
    3. Running the intervenable model's forward pass with these concatenated inputs
       and applying interventions at specified locations
    4. Extracting logits corresponding only to the positions where labels were appended
       (e.g., positions 9-11 in the example above)
    5. Computing accuracy and loss by comparing predicted continuations against ground truth
    
    This approach allows measuring how interventions affect the model's ability
    to predict the correct continuation, even for multi-token responses.
    
    Args:
        pipeline: The language model pipeline handling tokenization and generation
        intervenable_model: The model with intervention capabilities
        batch: Batch of data containing inputs and counterfactual inputs
        model_units_list: List of model units to intervene on
        
    Returns:
        tuple: (loss, eval_metrics, logging_info)
    """
    try:
        # Prepare intervenable inputs
        batched_base, batched_counterfactuals, inv_locations, feature_indices = _prepare_intervenable_inputs(
            pipeline, batch, model_units_list)

        # Get ground truth labels
        batched_inv_label = batch['label']

        if isinstance(batched_inv_label[0][0], dict): # this is a probability distribution
            all_mlabels, all_mprobs = [], []

            for bil in batched_inv_label:
                mlabels, mprobs = [], []
                for label, prob in bil[0].items():
                    if prob is not None:
                        mlabels.append(label)
                        mprobs.append(prob)

                mlabel_ids = pipeline.load(mlabels, add_special_tokens=False)["input_ids"][:,0]
                mprobs = torch.tensor(mprobs)

                all_mlabels.append(mlabel_ids)
                all_mprobs.append(mprobs)

            all_mlabels = torch.stack(all_mlabels)
            all_mprobs = torch.stack(all_mprobs).to("cuda")

            _, counterfactual_logits = intervenable_model(
                batched_base, batched_counterfactuals, unit_locations=inv_locations, subspaces=feature_indices)

            logits = counterfactual_logits.logits[:, -1]

            log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, vocab_size)
            log_probs_subset = torch.gather(log_probs, 1, all_mlabels)  # (batch_size, num_candidates)
            ce_per_batch = -(all_mprobs * log_probs_subset).sum(dim=-1)  # (batch_size,)
            loss = ce_per_batch.mean()

            return loss, {"accuracy": 0.0, "token_accuracy": 0.0}, {}

        batched_inv_label = pipeline.load(
            batched_inv_label, max_length=pipeline.max_new_tokens, padding_side='right', add_special_tokens=False)
        
        # Concatenate labels to base inputs for evaluation
        for k in batched_base:
            if isinstance(batched_base[k], torch.Tensor):
                batched_base[k] = torch.cat([batched_base[k], batched_inv_label[k]], dim=-1)
        
        # Run the intervenable model with interventions
        _, counterfactual_logits = intervenable_model(
            batched_base, batched_counterfactuals, unit_locations=inv_locations, subspaces=feature_indices)
        
        # Extract relevant portions of logits and labels for evaluation
        labels = batched_inv_label['input_ids']
        logits = counterfactual_logits.logits[:, -labels.shape[-1] - 1 : -1]
        pred_ids = torch.argmax(logits, dim=-1)
        
        # Compute metrics and loss
        eval_metrics = compute_metrics(pred_ids, labels, pipeline.tokenizer.pad_token_id)
        loss = compute_cross_entropy_loss(logits, labels, pipeline.tokenizer.pad_token_id)
        
        # Collect detailed information for logging
        logging_info = {
            "preds": pipeline.dump(pred_ids), 
            "labels": pipeline.dump(labels),
            "base_ids": batched_base["input_ids"][0],
            "base_masks": batched_base["attention_mask"][0],
            "counterfactual_masks": [c["attention_mask"][0] for c in batched_counterfactuals],
            "counterfactual_ids": [c["input_ids"][0] for c in batched_counterfactuals],
            "base_inputs": pipeline.dump(batched_base["input_ids"][0]),
            "counterfactual_inputs": [pipeline.dump(c["input_ids"][0]) for c in batched_counterfactuals],
            "inv_locations": inv_locations,
            "feature_indices": feature_indices
        }
        
        return loss, eval_metrics, logging_info
    except Exception as e:
        logger.error(f"Error in LM_loss_and_metric_fn: {str(e)}")
        raise

def compute_metrics(predicted_token_ids, eval_labels, pad_token_id):
    """
    Compute sequence-level and token-level accuracy metrics.
    
    Args:
        predicted_token_ids (torch.Tensor): Predicted token IDs from the model
        eval_labels (torch.Tensor): Ground truth token IDs 
        pad_token_id (int): ID of the padding token to be ignored in evaluation
    
    Returns:
        dict: Dictionary containing accuracy metrics:
            - accuracy: Proportion of sequences where all tokens match
            - token_accuracy: Proportion of individual tokens that match
    """
    try:
        # Create mask to ignore pad tokens in labels
        mask = (eval_labels != pad_token_id)

        # Calculate token-level accuracy (only for non-pad tokens)
        correct_tokens = (predicted_token_ids == eval_labels) & mask
        token_accuracy = correct_tokens.sum().float() / mask.sum() if mask.sum() > 0 else torch.tensor(1.0)

        # Calculate sequence-level accuracy (sequence correct if all non-pad tokens correct)
        sequence_correct = torch.stack([torch.all(correct_tokens[i, mask[i]]) for i in range(eval_labels.shape[0])])
        sequence_accuracy = sequence_correct.float().mean() if len(sequence_correct) > 0 else torch.tensor(1.0)

        return {
            "accuracy": float(sequence_accuracy.item()),
            "token_accuracy": float(token_accuracy.item())
        }
    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        return {"accuracy": 0.0, "token_accuracy": 0.0}

def compute_cross_entropy_loss(eval_preds, eval_labels, pad_token_id):
    """
    Compute cross-entropy loss over non-padding tokens.
    
    Args:
        eval_preds (torch.Tensor): Model predictions of shape (batch_size, seq_length, vocab_size)
        eval_labels (torch.Tensor): Ground truth labels of shape (batch_size, seq_length)
        pad_token_id (int): ID of the padding token to be ignored in loss calculation
    
    Returns:
        torch.Tensor: The computed cross-entropy loss
    """
    try:
        # Reshape predictions to (batch_size * sequence_length, vocab_size)
        batch_size, seq_length, vocab_size = eval_preds.shape
        preds_flat = eval_preds.reshape(-1, vocab_size)

        # Reshape labels to (batch_size * sequence_length)
        labels_flat = eval_labels.reshape(-1)

        # Create mask for non-pad tokens
        mask = labels_flat != pad_token_id

        # Only compute loss on non-pad tokens by filtering predictions and labels
        active_preds = preds_flat[mask]
        active_labels = labels_flat[mask]

        # Compute cross entropy loss
        loss = torch.nn.functional.cross_entropy(active_preds, active_labels)

        return loss
    except Exception as e:
        logger.error(f"Error computing loss: {str(e)}")
        return torch.tensor(0.0, requires_grad=True)


class PatchResidualStream(InterventionExperiment):
    """
    Experiment for analyzing residual stream interventions in language models.
    
    The residual stream is a fundamental concept in transformer architectures:
    - It represents the hidden representation that flows through the network
    - Each transformer layer adds its computation results to this stream
    - At any given layer L, the residual stream contains the sum of:
      * The original token embeddings
      * The outputs of all previous layers 0 to L-1
    
    This class enables interventions directly on the residual stream at specific points:
    - Layer index: Which transformer layer to target (0 to num_layers-1)
    - Token position: Which token in the sequence to modify
    
    By modifying the residual stream at strategic points and observing the effect on model outputs,
    we can identify where specific information is represented and how it's processed through
    the network. This approach is central to mechanistic interpretability, which aims to
    reverse-engineer the algorithms implemented by neural networks.
    
    Attributes:
        featurizers (Dict): Mapping of (layer, position) tuples to Featurizer instances
        loss_and_metric_fn (Callable): Function to compute loss and metrics
        layers (List[int]): Layer indices to analyze
        token_positions (List[TokenPosition]): Token positions to analyze
    """

    def __init__(self,
                 pipeline: LMPipeline,
                 causal_model: CausalModel,
                 layers: List[int],
                 token_positions: List[TokenPosition],
                 checker: Callable,
                 featurizers: Dict[Tuple[int, str], Featurizer] = None,
                 loss_and_metric_fn: Callable = LM_loss_and_metric_fn,
                 **kwargs):
        """
        Initialize ResidualStreamExperiment for analyzing residual stream interventions.
        
        Args:
            pipeline: LMPipeline object for model execution
            causal_model: CausalModel object for causal analysis
            layers: List of layer indices to analyze
            token_positions: List of ComponentIndexers for token positions
            checker: Function to evaluate output accuracy
            featurizers: Dict mapping (layer, position.id) to Featurizer instances
            **kwargs: Additional configuration options
        """
        self.featurizers = featurizers if featurizers is not None else {}
        self.loss_and_metric_fn = loss_and_metric_fn 

        # Generate all combinations of model units without feature_indices
        model_units_lists = []
        for layer in layers:
            for pos in token_positions:
                featurizer = self.featurizers.get((layer, pos.id), 
                                                 Featurizer(n_features=pipeline.model.config.hidden_size))
                model_units_lists.append([[
                    ResidualStream(
                        layer=layer,
                        token_indices=pos,
                        featurizer=featurizer,
                        shape=(pipeline.model.config.hidden_size,) if hasattr(pipeline.model.config, "hidden_size") else (pipeline.model.config.text_config.hidden_size,),
                        feature_indices=None, 
                        target_output=True
                    )
                ]])

        metadata_fn = lambda x: {"layer": x[0][0].component.get_layer(), 
                                "position": x[0][0].component.get_index_id()}

        super().__init__(
            pipeline=pipeline,
            causal_model=causal_model,
            model_units_lists=model_units_lists,
            checker=checker,
            metadata_fn=metadata_fn,
            **kwargs
        )
        
        self.layers = layers
        self.token_positions = token_positions

    def build_SAE_feature_intervention(self, sae_loader: Callable[[int], Any]) -> None:
        """
        Apply Sparse Autoencoder (SAE) features to model units.
        
        This method takes a function that loads SAEs for specific layers and 
        applies them to the appropriate model units. It handles memory cleanup 
        between loading SAEs for different layers to prevent OOM errors.
        
        Args:
            sae_loader: A function that takes a layer index and returns an SAE instance.
                For example:
                ```python
                def sae_loader(layer):
                    sae, _, _ = SAE.from_pretrained(
                        release = "gemma-scope-2b-pt-res-canonical",
                        sae_id = f"layer_{layer}/width_16k/canonical",
                        device = "cpu",
                    )
                    return sae
                ```
        
        Raises:
            RuntimeError: If SAE loading fails for a specific layer
        """
        try:
            # Process each model units list
            for model_units_list in self.model_units_lists:
                for model_units in model_units_list:
                    for unit in model_units:
                        layer = unit.component.get_layer()
                        
                        try:
                            # Load SAE for the specific layer
                            logger.info(f"Loading SAE for layer {layer}")
                            sae = sae_loader(layer)
                            
                            # Set the SAE featurizer for this unit
                            unit.set_featurizer(SAEFeaturizer(sae))
                            
                            # Clear GPU memory after loading each SAE
                            del sae
                            self._clean_memory()
                            
                        except Exception as e:
                            logger.error(f"Failed to load SAE for layer {layer}: {str(e)}")
                            # Continue with next unit rather than failing the entire experiment
                            continue
                            
            logger.info("Successfully applied SAE features to all model units")
            
        except Exception as e:
            logger.error(f"Error in build_SAE_feature_intervention: {str(e)}")
            raise RuntimeError(f"Failed to apply SAE features: {str(e)}")

    def _clean_memory(self):
        """
        Clean up memory to prevent OOM errors.
        
        This method performs garbage collection and clears CUDA cache
        to ensure memory is available for subsequent operations.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def plot_heatmaps(self, results: Dict, target_variables, save_path: str = None, average_counterfactuals: bool = False, title=None):
        """
        Generate heatmaps visualizing intervention scores across layers and positions.
        
        Args:
            results: Dictionary containing experiment results from interpret_results()
            target_variables: List of variable names being analyzed
            save_path: Optional path to save the generated plots. If None, displays plots interactively.
            average_counterfactuals: If True, averages scores across counterfactual datasets
        """
        target_variables_str = "-".join(target_variables)
        
        # Extract metadata from the first dataset for consistency
        metadata_map = self._extract_metadata_map(results)
        
        # Extract unique layers and positions from metadata
        layers, positions = self._extract_layers_positions(metadata_map)
        
        if average_counterfactuals:
            self._plot_average_heatmap(results, layers, positions, target_variables_str, save_path, title)
        else:
            self._plot_individual_heatmaps(results, layers, positions, target_variables_str, save_path, title)
    
    def _extract_metadata_map(self, results: Dict) -> Dict:
        """Extract metadata from the first dataset in results."""
        first_dataset = next(iter(results["dataset"]))
        metadata_map = {}
        
        for unit_str, unit_data in results["dataset"][first_dataset]["model_unit"].items():
            if "metadata" in unit_data:
                metadata_map[unit_str] = unit_data["metadata"]
                
        return metadata_map
    
    def _extract_layers_positions(self, metadata_map: Dict) -> Tuple[List, List]:
        """Extract unique layers and positions from metadata."""
        layers = sorted(list(set(
            metadata["layer"] for metadata in metadata_map.values() 
            if "layer" in metadata
        )), reverse=True)
        
        positions = list(set(
            metadata["position"] for metadata in metadata_map.values() 
            if "position" in metadata
        ))
        
        return layers, positions
    
    def _plot_average_heatmap(self, results: Dict, layers: List, positions: List, 
                             target_variables_str: str, save_path: Optional[str] = None, title=None):
        """Create and save/display an averaged heatmap across all datasets."""
        # Initialize score matrix and counter
        score_matrix = np.zeros((len(layers), len(positions)))
        dataset_count = 0.0
        
        # Sum scores across all datasets
        for dataset_name in results["dataset"]:
            temp_matrix = np.zeros((len(layers), len(positions)))
            valid_entries = False
            
            # Fill temporary matrix for this dataset
            for i, layer in enumerate(layers):
                for j, pos in enumerate(positions):
                    for unit_str, unit_data in results["dataset"][dataset_name]["model_unit"].items():
                        if "metadata" in unit_data and target_variables_str in unit_data:
                            if "average_score" in unit_data[target_variables_str]:
                                metadata = unit_data["metadata"]
                                if metadata.get("layer") == layer and metadata.get("position") == pos:
                                    temp_matrix[i, j] = unit_data[target_variables_str]["average_score"]
                                    valid_entries = True
            
            # Only include datasets with valid entries
            if valid_entries:
                score_matrix += temp_matrix
                dataset_count += 1
        
        # Calculate average across datasets
        if dataset_count > 0:
            score_matrix /= dataset_count
            
            # Create the heatmap
            self._create_heatmap(
                score_matrix=score_matrix,
                layers=layers,
                positions=positions,
                title=f'Intervention Accuracy - Average across {dataset_count} datasets\nTask: {results["task_name"]}' if not title else title,
                save_path=os.path.join(save_path, f'heatmap_average_{results["task_name"]}.png') if save_path else None
            )
        else:
            logger.warning("No valid data found for creating average heatmap")
    
    def _plot_individual_heatmaps(self, results: Dict, layers: List, positions: List, 
                                 target_variables_str: str, save_path: Optional[str] = None, title=None):
        """Create and save/display individual heatmaps for each dataset."""
        # Get dataset names
        dataset_names = list(results["dataset"].keys())
        
        # Track if we have valid data for any dataset
        any_valid_entries = False
        
        # Create individual heatmaps for each dataset
        for dataset_name in dataset_names:
            score_matrix = np.zeros((len(layers), len(positions)))
            valid_entries = False
            
            # Fill score matrix
            for i, layer in enumerate(layers):
                for j, pos in enumerate(positions):
                    for unit_str, unit_data in results["dataset"][dataset_name]["model_unit"].items():
                        if "metadata" in unit_data and target_variables_str in unit_data:
                            if "average_score" in unit_data[target_variables_str]:
                                metadata = unit_data["metadata"]
                                if metadata.get("layer") == layer and metadata.get("position") == pos:
                                    score_matrix[i, j] = unit_data[target_variables_str]["average_score"]
                                    valid_entries = True
            
            if valid_entries:
                any_valid_entries = True
                
                # Convert dataset name to a safe filename
                safe_dataset_name = dataset_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                
                # Create the heatmap
                self._create_heatmap(
                    score_matrix=score_matrix,
                    layers=layers,
                    positions=positions,
                    title=f'Intervention Accuracy - Dataset: {dataset_name}\nTask: {results["task_name"]}' if not title else title,
                    save_path=os.path.join(save_path, f'heatmap_{safe_dataset_name}_{results["task_name"]}.png') if save_path else None
                )
        
        if not any_valid_entries and save_path is None:
            logger.warning("No valid data found for visualization.")
    
    def _create_heatmap(self, score_matrix: np.ndarray, layers: List, positions: List, 
                       title: str, save_path: Optional[str] = None):
        """
        Create and save/display a single heatmap.
        
        Args:
            score_matrix: 2D numpy array with scores for each (layer, position) pair
            layers: List of layer indices
            positions: List of position names
            title: Title for the heatmap
            save_path: Path to save the heatmap, or None to display it
        """
        plt.figure(figsize=(3, 6))
        # plt.figure()
        display_matrix = np.round(score_matrix, 2)
        
        # Create the heatmap using seaborn
        sns.heatmap(
            score_matrix,
            xticklabels=positions,
            yticklabels=layers,
            cmap='viridis',
            annot=display_matrix,
            fmt='.2f',
            cbar_kws={'label': 'Accuracy (%)'},
            vmin=0,
            vmax=1,
        )
        
        plt.yticks(rotation=0)
        plt.xlabel('Position')
        plt.ylabel('Layer')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            # Create directory if it doesn't exist
            plt.show()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()