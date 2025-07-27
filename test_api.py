#!/usr/bin/env python3
"""
Test script for the LLM Fine-tuning API.
"""

import json
import time

import requests


def test_api_health():
    """Test if the API is running."""
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        print(f"✅ API is running: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ API is not running: {e}")
        return False

def test_prediction():
    """Test the prediction endpoint."""
    url = "http://localhost:8000/predict"
    headers = {
        "Authorization": "Bearer your-secret-token",
        "Content-Type": "application/json"
    }
    
    test_questions = [
        "What is Level 2 charging?",
        "How fast can DC fast chargers charge?",
        "What are the different connector types?",
        "How much does home charging cost?",
        "What is required to install a home charger?"
    ]
    
    print("🧪 Testing API predictions...")
    
    for i, question in enumerate(test_questions, 1):
        try:
            data = {"input_text": question}
            response = requests.post(url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Q{i}: {question}")
                print(f"   Response: {result.get('response', 'No response')}")
            else:
                print(f"❌ Q{i}: {question} - Status: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Q{i}: {question} - Request failed: {e}")
        
        time.sleep(1)  # Rate limiting

def test_authentication():
    """Test authentication."""
    url = "http://localhost:8000/predict"
    headers = {"Content-Type": "application/json"}
    data = {"input_text": "Test question"}
    
    print("🔐 Testing authentication...")
    
    # Test without token
    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)
        if response.status_code == 401:
            print("✅ Authentication working: 401 Unauthorized without token")
        else:
            print(f"⚠️ Unexpected response without token: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Authentication test failed: {e}")
    
    # Test with wrong token
    headers["Authorization"] = "Bearer wrong-token"
    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)
        if response.status_code == 401:
            print("✅ Authentication working: 401 Unauthorized with wrong token")
        else:
            print(f"⚠️ Unexpected response with wrong token: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Authentication test failed: {e}")

def main():
    """Run all API tests."""
    print("🧪 LLM FINE-TUNING API TEST")
    print("=" * 40)
    
    # Test 1: Health check
    if not test_api_health():
        print("❌ API is not running. Please start the pipeline first.")
        return
    
    print()
    
    # Test 2: Authentication
    test_authentication()
    
    print()
    
    # Test 3: Predictions
    test_prediction()
    
    print()
    print("🎉 API testing completed!")

if __name__ == "__main__":
    main() 