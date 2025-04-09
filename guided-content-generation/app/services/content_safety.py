import os
import base64
from typing import Dict, List, Optional
import httpx
import json

from app.core.config import settings


class ContentSafetyService:
    """Service for content moderation using Azure Content Safety API"""
    
    def __init__(self):
        self.endpoint = settings.CONTENT_SAFETY_ENDPOINT
        self.api_key = settings.CONTENT_SAFETY_KEY
        self.headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": self.api_key
        }
        # Mapping between severity levels and their numeric values
        self.severity_to_id = {"safe": 0, "low": 2, "med": 4, "high": 6}
        self.id_to_severity = {0: "safe", 2: "low", 4: "med", 6: "high"}
    
    async def analyze_image(self, image_path: str, thresholds: Dict[str, str]) -> Dict:
        """
        Analyze an image for harmful content using Azure Content Safety API
        
        Args:
            image_path: Path to the image file
            thresholds: Dictionary of category -> severity threshold
            
        Returns:
            Dictionary containing analysis results
        """
        # Convert thresholds to numeric values
        numeric_thresholds = {
            category: self.severity_to_id.get(severity.lower(), 2)
            for category, severity in thresholds.items()
        }
        
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            # Prepare request payload - Match the original implementation
            analyze_request = {
                "image": {
                    "content": base64_image
                },
                "categories": ["Hate", "SelfHarm", "Sexual", "Violence"],
                "outputType": "FourSeverityLevels"  # Match the original implementation
            }
            
            # Make API request - Match the original implementation's API version
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.endpoint}/contentsafety/image:analyze?api-version=2024-02-15-preview",
                    headers=self.headers,
                    json=analyze_request
                )
                
                if response.status_code != 200:
                    print(f"API request failed: {response.status_code} - {response.text}")
                    return {
                        "content_safety_results": {},
                        "error": f"API request failed with status {response.status_code}",
                        "details": response.text
                    }
                
                result = response.json()
                
                # Process results and apply thresholds - Match the original implementation's logic
                filtered_results = {}
                for category in result.get("categoriesAnalysis", []):
                    category_name = category.get("category")
                    severity = category.get("severity")
                    
                    # Store all results with filtering flag
                    filtered_results[category_name] = {
                        'filtered': severity >= numeric_thresholds.get(category_name, 2),
                        'severity': self.id_to_severity.get(severity, "unknown"),
                        'severity_score': severity,
                        'threshold': self.id_to_severity.get(numeric_thresholds.get(category_name, 2), "unknown")
                    }
                
                return {
                    "content_safety_results": filtered_results,
                    "raw_results": result
                }
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            # Return empty results but include the error
            return {
                "content_safety_results": {},
                "error": f"Error analyzing image: {str(e)}"
            }
    
    async def analyze_text(self, text: str, thresholds: Dict[str, str]) -> Dict:
        """
        Analyze text for harmful content using Azure Content Safety API
        
        Args:
            text: Text to analyze
            thresholds: Dictionary of category -> severity threshold
            
        Returns:
            Dictionary containing analysis results
        """
        # Convert thresholds to numeric values
        numeric_thresholds = {
            category: self.severity_to_id.get(severity.lower(), 2)
            for category, severity in thresholds.items()
        }
        
        try:
            # Prepare request payload - Match the original implementation
            analyze_request = {
                "text": text,
                "categories": ["Hate", "SelfHarm", "Sexual", "Violence"],
                "outputType": "FourSeverityLevels"  # Match the original implementation
            }
            
            # Make API request - Match the original implementation's API version
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.endpoint}/contentsafety/text:analyze?api-version=2024-02-15-preview",
                    headers=self.headers,
                    json=analyze_request
                )
                
                if response.status_code != 200:
                    print(f"API request failed: {response.status_code} - {response.text}")
                    return {
                        "content_safety_results": {},
                        "error": f"API request failed with status {response.status_code}",
                        "details": response.text
                    }
                
                result = response.json()
                
                # Process results and apply thresholds - Match the original implementation's logic
                filtered_results = {}
                for category in result.get("categoriesAnalysis", []):
                    category_name = category.get("category")
                    severity = category.get("severity")
                    
                    # Store all results with filtering flag
                    filtered_results[category_name] = {
                        'filtered': severity >= numeric_thresholds.get(category_name, 2),
                        'severity': self.id_to_severity.get(severity, "unknown"),
                        'severity_score': severity,
                        'threshold': self.id_to_severity.get(numeric_thresholds.get(category_name, 2), "unknown")
                    }
                
                return {
                    "content_safety_results": filtered_results,
                    "raw_results": result
                }
        except Exception as e:
            print(f"Error analyzing text: {str(e)}")
            # Return empty results but include the error
            return {
                "content_safety_results": {},
                "error": f"Error analyzing text: {str(e)}"
            }
