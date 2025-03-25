"""
Combined Parameter Extraction Tool

This module provides an integrated interface for extracting and querying both
dynamic parameters (from flight logs) and static parameters (from configuration files)
based on query intent detection.

Usage:
    from tools.combined_param_tool import CombinedParameterTool
    
    # Initialize the tool
    cp_tool = CombinedParameterTool(
        csv_dir="path/to/csv_topics", 
        kb_file="path/to/knowledge_base.json",
        static_csv="path/to/static_parameters.csv"
    )
    
    # Query for parameters
    results = cp_tool.query("What was the maximum speed during flight?")
    
    # Access the results
    for result in results:
        print(f"Parameter: {result['name']}")
        print(f"Type: {result['param_type']}")  # 'dynamic' or 'static'
        print(f"Score: {result['score']}")
        # Other fields depend on the parameter type
        print()
    
    # Or use the simplified function
    from tools.combined_param_tool import CP
    results = CP("What is the battery voltage?")
"""

import re
from typing import Dict, List, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Import the individual parameter tools
from tools.extract_dynamic_param_tool import DynamicParameterTool
from tools.extract_static_param_tool import StaticParameterTool

class CombinedParameterTool:
    """Tool for extracting and querying both dynamic and static parameters."""
    
    def __init__(self, csv_dir: str = "../csv_topics", 
                 kb_file: str = "../formatted_knowledge_base.json",
                 static_csv: str = "../static_parameters.csv"):
        """
        Initialize the Combined Parameter Tool.
        
        Args:
            csv_dir: Directory containing CSV files from flight logs
            kb_file: Path to the knowledge base JSON file
            static_csv: Path to the static parameters CSV file
        """
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize the individual tools
        self.dynamic_tool = DynamicParameterTool(csv_dir=csv_dir, kb_file=kb_file)
        self.static_tool = StaticParameterTool(csv_path=static_csv)
    
    def _analyze_query_intent(self, query: str) -> Dict[str, float]:
        """
        Analyze the query to determine if it's asking about dynamic data (measurements)
        or static parameters (configuration/limits).
        
        Returns:
            Dictionary with confidence scores for each intent type
        """
        query_lower = query.lower()
        
        # Patterns suggesting dynamic parameter query (measurements)
        dynamic_patterns = [
            r'\bwhat was\b', r'\bwhat were\b', r'\bwhat happened\b', r'\bduring\b', 
            r'\bmaximum\b.*\bduring\b', r'\bminimum\b.*\bduring\b',
            r'\bhighest\b.*\bduring\b', r'\blowest\b.*\bduring\b',
            r'\breached\b', r'\brecorded\b', r'\bmeasured\b', r'\bflight\b',
            r'\bactual\b', r'\btelemetry\b', r'\blog\b', r'\bdata\b'
        ]
        
        # Patterns suggesting static parameter query (configurations/limits)
        static_patterns = [
            r'\bwhat is\b', r'\bwhat are\b', r'\bhow is\b', r'\bhow are\b',
            r'\bconfigured\b', r'\bsetting\b', r'\bparameter\b', r'\bconfig\b',
            r'\blimit\b', r'\bconstraint\b', r'\brestriction\b', r'\bconfiguration\b',
            r'\ballowed\b', r'\bpermitted\b', r'\brequired\b', r'\ballowable\b'
        ]
        
        # Count pattern matches
        dynamic_score = sum(1 for pattern in dynamic_patterns if re.search(pattern, query_lower))
        static_score = sum(1 for pattern in static_patterns if re.search(pattern, query_lower))
        
        # Normalize scores
        max_score = max(dynamic_score, static_score, 1)  # Avoid division by zero
        dynamic_confidence = dynamic_score / max_score
        static_confidence = static_score / max_score
        
        # If neither type has a strong signal, use semantic similarity to typical questions
        if dynamic_score == 0 and static_score == 0:
            # Encode the query
            query_embedding = self.model.encode(query_lower)
            
            # Example queries for each type
            dynamic_examples = [
                "What was the maximum speed during flight?",
                "What altitude did the aircraft reach?",
                "What was the battery level at landing?",
                "What was the GPS position during takeoff?"
            ]
            
            static_examples = [
                "What is the maximum allowed speed?",
                "What is the minimum battery voltage?",
                "What are the altitude limits?",
                "What is the hover throttle setting?"
            ]
            
            # Encode examples
            dynamic_embeddings = self.model.encode([ex.lower() for ex in dynamic_examples])
            static_embeddings = self.model.encode([ex.lower() for ex in static_examples])
            
            # Calculate similarities
            def cosine_similarity(vec1, vec2):
                return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10))
            
            dynamic_similarities = [cosine_similarity(query_embedding, ex_emb) for ex_emb in dynamic_embeddings]
            static_similarities = [cosine_similarity(query_embedding, ex_emb) for ex_emb in static_embeddings]
            
            # Use max similarity for each type
            dynamic_confidence = max(dynamic_similarities)
            static_confidence = max(static_similarities)
        
        return {
            'dynamic': dynamic_confidence,
            'static': static_confidence
        }
    
    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query both dynamic and static parameters based on the query intent.
        
        Args:
            query_text: The natural language query
            top_k: Number of top results to return from each source
            
        Returns:
            List of result dictionaries with parameter information
        """
        # Analyze query intent
        intent = self._analyze_query_intent(query_text)
        
        results = []
        dynamic_results = []
        static_results = []
        
        # Get dynamic parameters if there's sufficient intent
        if intent['dynamic'] > 0.3:
            dynamic_results = self.dynamic_tool.query(query_text, top_k=top_k)
            
            # Add parameter type to results
            for result in dynamic_results:
                result['param_type'] = 'dynamic'
                
                # Adjust score based on intent confidence
                result['original_score'] = result['score']
                result['score'] = result['score'] * intent['dynamic']
            
            results.extend(dynamic_results)
        
        # Get static parameters if there's sufficient intent
        if intent['static'] > 0.3:
            static_results = self.static_tool.query(query_text, top_k=top_k)
            
            # Add parameter type to results
            for result in static_results:
                result['param_type'] = 'static'
                
                # Adjust score based on intent confidence
                result['original_score'] = result['score']
                result['score'] = result['score'] * intent['static']
            
            results.extend(static_results)
        
        # If we didn't get any results with our intent threshold, try both
        if not results:
            dynamic_results = self.dynamic_tool.query(query_text, top_k=top_k)
            for result in dynamic_results:
                result['param_type'] = 'dynamic'
                result['original_score'] = result['score']
                result['score'] = result['score'] * 0.5  # Neutral confidence
            
            static_results = self.static_tool.query(query_text, top_k=top_k)
            for result in static_results:
                result['param_type'] = 'static'
                result['original_score'] = result['score']
                result['score'] = result['score'] * 0.5  # Neutral confidence
            
            results = dynamic_results + static_results
        
        # Sort by adjusted score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Return up to top_k overall results
        return results[:top_k]
    
    def query_dynamic(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Explicitly query only dynamic parameters."""
        results = self.dynamic_tool.query(query_text, top_k=top_k)
        for result in results:
            result['param_type'] = 'dynamic'
        return results
    
    def query_static(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Explicitly query only static parameters."""
        results = self.static_tool.query(query_text, top_k=top_k)
        for result in results:
            result['param_type'] = 'static'
        return results

def CP(query: str) -> List[Dict[str, Any]]:
    """
    Utility function to search for both dynamic and static parameters based on a query.
    
    This is a convenience function that creates a CombinedParameterTool instance
    with default paths or attempts to find the files in the current directory.
    
    Args:
        query: The query text to search for parameters
        
    Returns:
        A list of matching parameters with their details
    """
    # Try various possible locations for the required files
    
    # Default relative paths
    default_csv_dir = "../csv_topics"
    default_kb_file = "../formatted_knowledge_base.json"
    default_static_csv = "../static_parameters.csv"
    
    # Current directory paths
    current_csv_dir = "csv_topics"
    current_kb_file = "formatted_knowledge_base.json"
    current_static_csv = "static_parameters.csv"
    
    # Environment paths if available
    env_csv_dir = os.environ.get("CSV_TOPICS_DIR")
    env_kb_file = os.environ.get("KNOWLEDGE_BASE_FILE")
    env_static_csv = os.environ.get("STATIC_PARAMS_PATH")
    
    # Find the first existing CSV directory
    csv_dir = None
    for dir_path in [env_csv_dir, default_csv_dir, current_csv_dir]:
        if dir_path and os.path.exists(dir_path) and os.path.isdir(dir_path):
            csv_dir = dir_path
            break
    
    # Find the first existing knowledge base file
    kb_file = None
    for file_path in [env_kb_file, default_kb_file, current_kb_file]:
        if file_path and os.path.exists(file_path):
            kb_file = file_path
            break
    
    # Find the first existing static parameters CSV
    static_csv = None
    for file_path in [env_static_csv, default_static_csv, current_static_csv]:
        if file_path and os.path.exists(file_path):
            static_csv = file_path
            break
    
    if not csv_dir:
        print("Warning: csv_topics directory not found in any location")
        return []
    
    if not kb_file:
        print("Warning: formatted_knowledge_base.json not found in any location")
        # Still proceed as the tool can build a knowledge base from CSV files
    
    if not static_csv:
        print("Warning: static_parameters.csv not found in any location")
        return []
    
    # Initialize the tool with the found paths
    print(f"Using CSV directory: {csv_dir}")
    print(f"Using knowledge base file: {kb_file or 'None (will build from scratch)'}")
    print(f"Using static parameters file: {static_csv}")
    
    tool = CombinedParameterTool(
        csv_dir=csv_dir, 
        kb_file=kb_file or "formatted_knowledge_base.json",
        static_csv=static_csv
    )
    
    # Query and return results
    return tool.query(query)

# Example usage
if __name__ == "__main__":
    # Examples of both dynamic and static queries
    examples = [
        "What was the maximum speed during the flight?",  # Dynamic
        "What is the maximum allowed speed?",             # Static
        "What was the battery voltage at landing?",       # Dynamic
        "What is the minimum battery voltage allowed?"    # Static
    ]
    
    tool = CombinedParameterTool()
    
    for example in examples:
        print("\n" + "=" * 60)
        print(f"Query: {example}")
        intent = tool._analyze_query_intent(example)
        print(f"Intent: Dynamic {intent['dynamic']:.2f}, Static {intent['static']:.2f}")
        
        results = tool.query(example, top_k=3)
        print(f"Top results:")
        
        for i, result in enumerate(results, 1):
            param_type = result['param_type']
            print(f"{i}. [{param_type.upper()}] {result.get('name', 'Unknown')}")
            print(f"   Score: {result['score']:.4f} (Original: {result['original_score']:.4f})")
            
            if param_type == 'dynamic':
                print(f"   Description: {result.get('description', 'None')}")
                print(f"   Key: {result.get('key', 'None')}")
                print(f"   Fields: {len(result.get('fields', {}))} fields available")
            else:  # static
                print(f"   Group: {result.get('group', 'None')}")
                print(f"   Value: {result.get('value', 'None')} {result.get('unit', '')}")
                print(f"   Description: {result.get('short_desc', 'None')}")
                
                if result.get('min', '') or result.get('max', ''):
                    limits = []
                    if result.get('min', '') != '':
                        limits.append(f"Min: {result['min']}")
                    if result.get('max', '') != '':
                        limits.append(f"Max: {result['max']}")
                    print(f"   Limits: {', '.join(limits)}")
            print() 