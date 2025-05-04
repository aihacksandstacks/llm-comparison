"""
Copyright (c) 2025 AI Hacks and Stacks
All rights reserved.

This file is part of the LLM Comparison Tool.

Tests for the evaluation module.
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from src.features.llm_compare.evaluation import EvaluationManager, get_evaluation_manager


@pytest.fixture
def mock_eval_data_dir(tmp_path):
    """Create a temporary directory for evaluation data."""
    eval_dir = tmp_path / "data" / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir

@pytest.fixture
def mock_eval_config():
    """Mock evaluation configuration for testing."""
    return {
        "metrics": [
            {"name": "rouge", "parameters": {"rouge_types": ["rouge1", "rouge2", "rougeL"]}},
            {"name": "semantic_similarity", "parameters": {"model": "nomic-embed-text-v1.5"}},
            {"name": "response_time"},
            {"name": "token_count"}
        ]
    }

@pytest.fixture
def mock_experiment():
    """Create a mock experiment for testing."""
    return {
        "id": "test-experiment-id",
        "name": "Test Experiment",
        "tags": ["test", "llm-comparison"],
        "timestamp": 1622000000.0,
        "metrics": [
            {"name": "rouge"},
            {"name": "semantic_similarity"}
        ],
        "samples": []
    }


class TestEvaluationManager:
    """Tests for the EvaluationManager class."""
    
    @patch("src.features.llm_compare.evaluation.get_config")
    @patch("src.features.llm_compare.evaluation.COMET_API_KEY", "test_api_key")
    @patch("src.features.llm_compare.evaluation.COMET_CONFIG", {"workspace": "test-workspace", "project_name": "test-project"})
    @patch("src.features.llm_compare.evaluation.DATA_DIR")
    def test_init(self, mock_data_dir, mock_get_config, mock_eval_config, mock_eval_data_dir):
        """Test initialization of EvaluationManager."""
        # Setup
        mock_data_dir.return_value = str(mock_eval_data_dir.parent)
        mock_get_config.return_value = mock_eval_config
        
        # Execute
        with patch("src.features.llm_compare.evaluation.EvaluationManager._setup_comet_ml"):
            manager = EvaluationManager()
        
        # Assert
        mock_get_config.assert_called_once_with("evaluation")
        assert manager.metrics == mock_eval_config["metrics"]
        assert manager.api_key == "test_api_key"
        assert manager.workspace == "test-workspace"
        assert manager.project_name == "test-project"
    
    @patch("src.features.llm_compare.evaluation.get_config")
    @patch("src.features.llm_compare.evaluation.COMET_API_KEY", "")
    @patch("src.features.llm_compare.evaluation.DATA_DIR")
    def test_init_no_api_key(self, mock_data_dir, mock_get_config, mock_eval_config, mock_eval_data_dir):
        """Test initialization without API key."""
        # Setup
        mock_data_dir.return_value = str(mock_eval_data_dir.parent)
        mock_get_config.return_value = mock_eval_config
        
        # Execute
        manager = EvaluationManager()
        
        # Assert
        assert manager.api_key == ""
        assert manager.comet_ml is None
        assert manager.opik is None
    
    @patch("src.features.llm_compare.evaluation.get_config")
    @patch("src.features.llm_compare.evaluation.COMET_API_KEY", "test_api_key")
    @patch("src.features.llm_compare.evaluation.COMET_CONFIG", {"workspace": "test-workspace", "project_name": "test-project"})
    @patch("src.features.llm_compare.evaluation.DATA_DIR")
    @patch("src.features.llm_compare.evaluation.uuid.uuid4")
    @patch("src.features.llm_compare.evaluation.time.time")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_create_experiment(self, mock_json_dump, mock_file, mock_time, mock_uuid, 
                             mock_data_dir, mock_get_config, mock_eval_config, mock_eval_data_dir):
        """Test creating an experiment."""
        # Setup
        mock_data_dir.return_value = str(mock_eval_data_dir.parent)
        mock_get_config.return_value = mock_eval_config
        mock_time.return_value = 1622000000.0
        mock_uuid.return_value = "test-experiment-id"
        
        # Execute
        with patch("src.features.llm_compare.evaluation.EvaluationManager._setup_comet_ml"):
            manager = EvaluationManager()
            experiment = manager.create_experiment(name="Test Experiment", tags=["test"])
        
        # Assert
        assert experiment["id"] == "test-experiment-id"
        assert experiment["name"] == "Test Experiment"
        assert "test" in experiment["tags"]
        assert "llm-comparison" in experiment["tags"]
        assert experiment["timestamp"] == 1622000000.0
        assert len(experiment["metrics"]) == 4
        assert len(experiment["samples"]) == 0
        
        mock_file.assert_called_once()
        mock_json_dump.assert_called_once()
    
    @patch("src.features.llm_compare.evaluation.get_config")
    @patch("src.features.llm_compare.evaluation.COMET_API_KEY", "test_api_key")
    @patch("src.features.llm_compare.evaluation.DATA_DIR")
    @patch("src.features.llm_compare.evaluation.uuid.uuid4")
    @patch("src.features.llm_compare.evaluation.time.time")
    @patch("builtins.open", new_callable=mock_open, read_data='{"id": "test-experiment-id", "samples": []}')
    @patch("json.dump")
    @patch("json.load")
    def test_log_prompt(self, mock_json_load, mock_json_dump, mock_file, mock_time, mock_uuid,
                      mock_data_dir, mock_get_config, mock_eval_config, mock_experiment):
        """Test logging a prompt to an experiment."""
        # Setup
        mock_data_dir.return_value = "data"
        mock_get_config.return_value = mock_eval_config
        mock_time.return_value = 1622000100.0
        mock_uuid.return_value = "test-sample-id"
        mock_json_load.return_value = {"id": "test-experiment-id", "samples": []}
        
        # Execute
        with patch("src.features.llm_compare.evaluation.EvaluationManager._setup_comet_ml"):
            manager = EvaluationManager()
            sample = manager.log_prompt(
                experiment=mock_experiment,
                prompt="What is RAG?",
                system_prompt="You are a helpful assistant.",
                context="RAG stands for Retrieval-Augmented Generation."
            )
        
        # Assert
        assert sample["id"] == "test-sample-id"
        assert sample["prompt"] == "What is RAG?"
        assert sample["system_prompt"] == "You are a helpful assistant."
        assert sample["context"] == "RAG stands for Retrieval-Augmented Generation."
        assert sample["timestamp"] == 1622000100.0
        assert len(sample["responses"]) == 0
        
        assert mock_file.call_count == 2  # One read, one write
        assert mock_json_dump.call_count == 1
        assert mock_json_load.call_count == 1
        
        # Check if sample was added to the experiment
        assert len(mock_experiment["samples"]) == 1
        assert mock_experiment["samples"][0] == sample
    
    @patch("src.features.llm_compare.evaluation.get_config")
    @patch("src.features.llm_compare.evaluation.COMET_API_KEY", "test_api_key")
    @patch("src.features.llm_compare.evaluation.DATA_DIR")
    @patch("src.features.llm_compare.evaluation.uuid.uuid4")
    @patch("src.features.llm_compare.evaluation.time.time")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("json.load")
    def test_log_response(self, mock_json_load, mock_json_dump, mock_file, mock_time, mock_uuid,
                        mock_data_dir, mock_get_config, mock_eval_config):
        """Test logging a response to a prompt."""
        # Setup
        mock_data_dir.return_value = "data"
        mock_get_config.return_value = mock_eval_config
        mock_time.return_value = 1622000200.0
        mock_uuid.return_value = "test-response-id"
        
        # Create a sample experiment with a sample
        experiment = {
            "id": "test-experiment-id",
            "name": "Test Experiment",
            "samples": [
                {
                    "id": "test-sample-id",
                    "prompt": "What is RAG?",
                    "system_prompt": "You are a helpful assistant.",
                    "context": "RAG stands for Retrieval-Augmented Generation.",
                    "responses": []
                }
            ]
        }
        
        mock_json_load.return_value = {
            "id": "test-experiment-id",
            "samples": [
                {
                    "id": "test-sample-id",
                    "prompt": "What is RAG?",
                    "system_prompt": "You are a helpful assistant.",
                    "context": "RAG stands for Retrieval-Augmented Generation.",
                    "responses": []
                }
            ]
        }
        
        # Mock response
        response = {
            "text": "RAG is Retrieval-Augmented Generation, a technique that combines retrieval with generation.",
            "model": "gpt-4",
            "provider": "openai",
            "metadata": {"response_time": 0.5},
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }
        
        # Mock ground truth
        ground_truth = "RAG (Retrieval-Augmented Generation) is a technique that enhances LLM outputs with external knowledge."
        
        # Execute
        with patch("src.features.llm_compare.evaluation.EvaluationManager._setup_comet_ml"):
            with patch("src.features.llm_compare.evaluation.EvaluationManager._calculate_metrics") as mock_calc_metrics:
                mock_calc_metrics.return_value = {"rouge1_f": 0.8, "semantic_similarity": 0.9}
                
                manager = EvaluationManager()
                response_data = manager.log_response(
                    experiment=experiment,
                    sample_id="test-sample-id",
                    response=response,
                    ground_truth=ground_truth
                )
        
        # Assert
        assert response_data["id"] == "test-response-id"
        assert response_data["text"] == response["text"]
        assert response_data["model"] == "gpt-4"
        assert response_data["provider"] == "openai"
        assert response_data["timestamp"] == 1622000200.0
        assert response_data["ground_truth"] == ground_truth
        assert response_data["metrics"] == {"rouge1_f": 0.8, "semantic_similarity": 0.9}
        
        assert mock_file.call_count == 2  # One read, one write
        assert mock_json_dump.call_count == 1
        assert mock_json_load.call_count == 1
        
        # Check if response was added to the sample
        assert len(experiment["samples"][0]["responses"]) == 1
        assert experiment["samples"][0]["responses"][0] == response_data
    
    @patch("src.features.llm_compare.evaluation.get_evaluation_manager")
    def test_get_evaluation_manager(self, mock_get_evaluation_manager):
        """Test the get_evaluation_manager factory function."""
        # Setup
        mock_manager = MagicMock()
        mock_get_evaluation_manager.return_value = mock_manager
        
        # Execute
        result = get_evaluation_manager()
        
        # Assert
        assert result == mock_manager 