"""
Evaluation module for the LLM Comparison Tool.
Provides functionality for benchmarking and evaluating LLM outputs using Comet ML Opik.
"""

import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
import uuid

from src.shared.config import DATA_DIR, COMET_API_KEY, COMET_CONFIG, get_config

class EvaluationManager:
    """Class for handling LLM evaluation and benchmarking with Comet ML Opik."""
    
    def __init__(self):
        """Initialize the evaluation manager."""
        # Get evaluation configuration
        eval_config = get_config("evaluation")
        self.metrics = eval_config.get("metrics", [])
        
        # Set up API key for Comet ML
        self.api_key = COMET_API_KEY
        self.workspace = COMET_CONFIG.get("workspace", "")
        self.project_name = COMET_CONFIG.get("project_name", "llm_comparison")
        
        if not self.api_key:
            print("Warning: Comet ML API key not found. Evaluation will be limited to local metrics only.")
        
        # Create data directory for evaluation results
        self.eval_data_dir = Path(DATA_DIR) / "evaluations"
        self.eval_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up Comet ML client if API key is available
        self.comet_ml = None
        self.opik = None
        if self.api_key:
            self._setup_comet_ml()
    
    def _setup_comet_ml(self):
        """Set up Comet ML client and Opik for evaluation."""
        try:
            import comet_ml
            from comet_ml.opik import Opik
            
            # Initialize Comet ML with API key
            comet_ml.init(
                api_key=self.api_key,
                workspace=self.workspace,
                project_name=self.project_name
            )
            
            self.comet_ml = comet_ml
            self.opik = Opik()
            
            print(f"Initialized Comet ML Opik for workspace {self.workspace}, project {self.project_name}")
        
        except ImportError:
            print("Comet ML package not installed. Please install it with 'pip install comet-ml'.")
            self.comet_ml = None
            self.opik = None
    
    def create_experiment(self, name: Optional[str] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a new evaluation experiment.
        
        Args:
            name: Optional name for the experiment.
            tags: Optional list of tags for the experiment.
            
        Returns:
            Dictionary with experiment metadata.
        """
        # Generate a default name if none provided
        if not name:
            name = f"llm-comparison-{time.strftime('%Y%m%d-%H%M%S')}"
        
        # Set up tags
        if not tags:
            tags = ["llm-comparison"]
        
        # Generate experiment ID
        experiment_id = str(uuid.uuid4())
        
        # Create experiment metadata
        experiment = {
            "id": experiment_id,
            "name": name,
            "tags": tags,
            "timestamp": time.time(),
            "metrics": self.metrics,
            "samples": []
        }
        
        # Create Comet experiment if available
        if self.comet_ml:
            comet_experiment = self.comet_ml.Experiment(
                project_name=self.project_name,
                workspace=self.workspace,
                api_key=self.api_key
            )
            
            comet_experiment.set_name(name)
            for tag in tags:
                comet_experiment.add_tag(tag)
            
            experiment["comet_experiment"] = comet_experiment
        
        # Save experiment metadata
        experiment_path = self.eval_data_dir / f"{experiment_id}.json"
        with open(experiment_path, "w") as f:
            # Create a copy without the comet_experiment object
            save_exp = experiment.copy()
            if "comet_experiment" in save_exp:
                del save_exp["comet_experiment"]
            json.dump(save_exp, f, indent=2)
        
        print(f"Created experiment: {name} (ID: {experiment_id})")
        return experiment
    
    def log_prompt(self, experiment: Dict[str, Any], 
                  prompt: str, 
                  system_prompt: Optional[str] = None,
                  context: Optional[str] = None,
                  sample_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Log a prompt to the experiment.
        
        Args:
            experiment: Experiment dictionary.
            prompt: User prompt to log.
            system_prompt: Optional system prompt.
            context: Optional context (like RAG context).
            sample_id: Optional sample ID (generated if not provided).
            
        Returns:
            Dictionary with sample metadata.
        """
        # Generate sample ID if not provided
        if not sample_id:
            sample_id = str(uuid.uuid4())
        
        # Create sample metadata
        sample = {
            "id": sample_id,
            "timestamp": time.time(),
            "prompt": prompt,
            "system_prompt": system_prompt,
            "context": context,
            "responses": []
        }
        
        # Log to Comet ML if available
        if "comet_experiment" in experiment and self.opik:
            comet_experiment = experiment["comet_experiment"]
            
            # Create a text input for tracking
            self.opik.log_text_input(
                prompt=prompt,
                system_prompt=system_prompt if system_prompt else "",
                context=context if context else "",
                input_id=sample_id
            )
            
            # Log input metadata
            comet_experiment.log_parameter(f"prompt_{sample_id}", prompt)
            if system_prompt:
                comet_experiment.log_parameter(f"system_prompt_{sample_id}", system_prompt)
            if context:
                comet_experiment.log_parameter(f"context_{sample_id}", context)
        
        # Add to experiment samples
        experiment["samples"].append(sample)
        
        # Update experiment file
        experiment_path = self.eval_data_dir / f"{experiment['id']}.json"
        with open(experiment_path, "r") as f:
            saved_exp = json.load(f)
        
        saved_exp["samples"].append({
            "id": sample_id,
            "timestamp": sample["timestamp"],
            "prompt": prompt,
            "system_prompt": system_prompt,
            "context": context,
            "responses": []
        })
        
        with open(experiment_path, "w") as f:
            json.dump(saved_exp, f, indent=2)
        
        print(f"Logged prompt with ID {sample_id} to experiment {experiment['name']}")
        return sample
    
    def log_response(self, experiment: Dict[str, Any], 
                    sample_id: str,
                    response: Dict[str, Any],
                    ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Log an LLM response to a prompt.
        
        Args:
            experiment: Experiment dictionary.
            sample_id: ID of the sample/prompt.
            response: Response dictionary with text, model, etc.
            ground_truth: Optional ground truth for evaluation.
            
        Returns:
            Dictionary with response evaluation metrics.
        """
        # Find the sample in the experiment
        sample = None
        for s in experiment["samples"]:
            if s["id"] == sample_id:
                sample = s
                break
        
        if not sample:
            raise ValueError(f"Sample with ID {sample_id} not found in experiment {experiment['id']}")
        
        # Extract response text and metadata
        response_text = response.get("text", "")
        model = response.get("model", "unknown")
        provider = response.get("provider", "unknown")
        
        # Create response metadata
        response_data = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "text": response_text,
            "model": model,
            "provider": provider,
            "metadata": response.get("metadata", {}),
            "ground_truth": ground_truth,
            "metrics": {}
        }
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            response_text=response_text,
            prompt=sample["prompt"],
            ground_truth=ground_truth,
            context=sample.get("context")
        )
        
        response_data["metrics"] = metrics
        
        # Log to Comet ML if available
        if "comet_experiment" in experiment and self.opik:
            comet_experiment = experiment["comet_experiment"]
            
            # Log the model response
            self.opik.log_model_response(
                response=response_text,
                model=model,
                provider=provider,
                input_id=sample_id,
                response_id=response_data["id"]
            )
            
            # If ground truth is available, log for evaluation
            if ground_truth:
                self.opik.log_ground_truth(
                    ground_truth=ground_truth,
                    input_id=sample_id
                )
            
            # Log metrics to Comet
            for metric_name, metric_value in metrics.items():
                comet_experiment.log_metric(
                    f"{metric_name}_{model}_{sample_id}", 
                    metric_value
                )
            
            # Log usage information if available
            if "usage" in response:
                for usage_key, usage_value in response["usage"].items():
                    comet_experiment.log_metric(
                        f"{usage_key}_{model}_{sample_id}",
                        usage_value
                    )
        
        # Add to sample responses
        sample["responses"].append(response_data)
        
        # Update experiment file
        experiment_path = self.eval_data_dir / f"{experiment['id']}.json"
        with open(experiment_path, "r") as f:
            saved_exp = json.load(f)
        
        # Find the sample in the saved experiment
        for saved_sample in saved_exp["samples"]:
            if saved_sample["id"] == sample_id:
                saved_sample["responses"].append(response_data)
                break
        
        with open(experiment_path, "w") as f:
            json.dump(saved_exp, f, indent=2)
        
        print(f"Logged response from {model} ({provider}) for prompt {sample_id}")
        return response_data
    
    def _calculate_metrics(self, response_text: str, prompt: str, 
                         ground_truth: Optional[str] = None,
                         context: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate evaluation metrics for a response.
        
        Args:
            response_text: LLM response text.
            prompt: Original prompt.
            ground_truth: Optional ground truth for comparison.
            context: Optional context from RAG.
            
        Returns:
            Dictionary with metric names and values.
        """
        metrics = {}
        
        # Calculate metrics based on configuration
        for metric_config in self.metrics:
            metric_name = metric_config.get("name", "")
            
            if not metric_name:
                continue
                
            # Calculate different metrics based on name
            if metric_name == "rouge" and ground_truth:
                metrics.update(self._calculate_rouge(response_text, ground_truth))
            
            elif metric_name == "semantic_similarity" and ground_truth:
                metrics["semantic_similarity"] = self._calculate_semantic_similarity(
                    response_text, ground_truth
                )
            
            elif metric_name == "response_time" and self.opik:
                # This would be handled by Comet ML's tracking
                pass
            
            elif metric_name == "token_count" and self.opik:
                # This would be handled by Comet ML's tracking
                pass
        
        return metrics
    
    def _calculate_rouge(self, response_text: str, ground_truth: str) -> Dict[str, float]:
        """Calculate ROUGE metrics between response and ground truth."""
        try:
            from rouge import Rouge
            
            rouge = Rouge()
            scores = rouge.get_scores(response_text, ground_truth)[0]
            
            # Extract and flatten scores
            metrics = {}
            for rouge_type, scores in scores.items():
                for score_type, value in scores.items():
                    metrics[f"{rouge_type}_{score_type}"] = value
            
            return metrics
        
        except ImportError:
            print("Rouge package not installed. ROUGE metrics will not be calculated.")
            return {"rouge_error": 0.0}
    
    def _calculate_semantic_similarity(self, response_text: str, ground_truth: str) -> float:
        """Calculate semantic similarity between response and ground truth."""
        from src.features.llm_compare.embeddings import get_embedding_provider
        
        try:
            # Get embedding provider
            embedding_provider = get_embedding_provider()
            
            # Generate embeddings
            response_embedding = embedding_provider.get_embedding(response_text)
            truth_embedding = embedding_provider.get_embedding(ground_truth)
            
            # Calculate cosine similarity
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            response_vec = np.array(response_embedding).reshape(1, -1)
            truth_vec = np.array(truth_embedding).reshape(1, -1)
            
            similarity = cosine_similarity(response_vec, truth_vec)[0][0]
            return float(similarity)
        
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def load_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Load an existing experiment by ID.
        
        Args:
            experiment_id: ID of the experiment to load.
            
        Returns:
            Experiment dictionary or None if not found.
        """
        experiment_path = self.eval_data_dir / f"{experiment_id}.json"
        
        if not experiment_path.exists():
            print(f"Experiment with ID {experiment_id} not found")
            return None
        
        try:
            with open(experiment_path, "r") as f:
                experiment = json.load(f)
            
            # Create Comet experiment if API key is available
            if self.comet_ml:
                comet_experiment = self.comet_ml.ExistingExperiment(
                    api_key=self.api_key,
                    experiment_key=experiment_id,
                    project_name=self.project_name,
                    workspace=self.workspace
                )
                
                experiment["comet_experiment"] = comet_experiment
            
            print(f"Loaded experiment: {experiment['name']} (ID: {experiment_id})")
            return experiment
        
        except Exception as e:
            print(f"Error loading experiment {experiment_id}: {e}")
            return None


def get_evaluation_manager() -> EvaluationManager:
    """
    Factory function to get an EvaluationManager instance.
    
    Returns:
        EvaluationManager instance.
    """
    return EvaluationManager() 