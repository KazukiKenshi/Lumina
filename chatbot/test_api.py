"""
Test script for Mental Health Chatbot API
Tests all endpoints and displays responses with formatting
"""

import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:5001"
HEADERS = {"Content-Type": "application/json"}

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_response(response, show_full=True):
    """Print formatted API response"""
    print(f"\nStatus Code: {response.status_code}")
    try:
        data = response.json()
        if show_full:
            print(f"Response:\n{json.dumps(data, indent=2)}")
        else:
            # Show abbreviated response
            if 'text' in data:
                text = data['text']
                print(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
            if 'emotion' in data:
                print(f"Emotion: {data['emotion']}")
            if 'session_id' in data:
                print(f"Session ID: {data['session_id']}")
    except:
        print(f"Response: {response.text}")

def test_health_check():
    """Test the health check endpoint"""
    print_section("1. Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print_response(response)
    return response.status_code == 200

def test_chat_new_session():
    """Test chat endpoint with a new session"""
    print_section("2. Chat - New Session")
    
    payload = {
        "message": "I've been feeling really tired and sad for the past few weeks",
        "user_id": "test_user_123"
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    response = requests.post(f"{BASE_URL}/api/chat", json=payload, headers=HEADERS)
    print_response(response, show_full=False)
    
    if response.status_code == 200:
        data = response.json()
        return data.get('session_id')
    return None

def test_chat_existing_session(session_id):
    """Test chat endpoint with an existing session"""
    print_section("3. Chat - Existing Session")
    
    payload = {
        "message": "What can I do to feel better?",
        "session_id": session_id,
        "user_id": "test_user_123"
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    response = requests.post(f"{BASE_URL}/api/chat", json=payload, headers=HEADERS)
    print_response(response, show_full=False)
    
    return response.status_code == 200

def test_multiple_messages(session_id):
    """Test multiple messages in sequence"""
    print_section("4. Chat - Multiple Messages")
    
    messages = [
        "I can't sleep at night",
        "Do you think this could be depression?",
        "How long does it usually take to feel better?"
    ]
    
    for i, msg in enumerate(messages, 1):
        print(f"\n--- Message {i} ---")
        payload = {
            "message": msg,
            "session_id": session_id,
            "user_id": "test_user_123"
        }
        
        print(f"User: {msg}")
        response = requests.post(f"{BASE_URL}/api/chat", json=payload, headers=HEADERS)
        
        if response.status_code == 200:
            data = response.json()
            print(f"Assistant: {data['text'][:150]}...")
            print(f"Emotion: {data['emotion']}")
        else:
            print(f"Error: {response.status_code}")
        
        time.sleep(1)  # Avoid rate limiting

def test_session_summary(session_id):
    """Test session summary endpoint"""
    print_section("5. Session Summary")
    
    response = requests.get(f"{BASE_URL}/api/session/{session_id}/summary")
    print_response(response)
    
    return response.status_code == 200

def test_conversation_history(session_id):
    """Test conversation history endpoint"""
    print_section("6. Conversation History")
    
    response = requests.get(f"{BASE_URL}/api/session/{session_id}/history")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nSession ID: {data['session_id']}")
        print(f"Messages: {len(data['conversation_history'])}")
        
        # Show first few messages
        print("\nFirst 4 messages:")
        for msg in data['conversation_history'][:4]:
            role = msg['role'].upper()
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            print(f"  [{role}] {content}")
    else:
        print_response(response)
    
    return response.status_code == 200

def test_list_sessions():
    """Test list all sessions endpoint"""
    print_section("7. List All Sessions")
    
    response = requests.get(f"{BASE_URL}/api/sessions")
    print_response(response)
    
    return response.status_code == 200

def test_invalid_request():
    """Test error handling with invalid request"""
    print_section("8. Error Handling - Invalid Request")
    
    payload = {
        "invalid_field": "no message field"
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    response = requests.post(f"{BASE_URL}/api/chat", json=payload, headers=HEADERS)
    print_response(response)
    
    return response.status_code == 400

def test_delete_session(session_id):
    """Test delete session endpoint"""
    print_section("9. Delete Session")
    
    response = requests.delete(f"{BASE_URL}/api/session/{session_id}")
    print_response(response)
    
    # Verify deletion
    print("\nVerifying deletion...")
    verify_response = requests.get(f"{BASE_URL}/api/session/{session_id}/summary")
    print(f"Verification Status: {verify_response.status_code} (should be 404)")
    
    return response.status_code == 200 and verify_response.status_code == 404

def test_emotion_variety():
    """Test different emotions based on message tone"""
    print_section("10. Emotion Variety Test")
    
    test_cases = [
        ("Tell me about healthy sleep habits", "Expected: neutral or happy"),
        ("I'm struggling with panic attacks", "Expected: sad"),
        ("I've been doing better with my therapy exercises", "Expected: happy"),
    ]
    
    for message, expected in test_cases:
        print(f"\n--- Test Case ---")
        print(f"Message: {message}")
        print(f"{expected}")
        
        payload = {
            "message": message,
            "user_id": "emotion_test"
        }
        
        response = requests.post(f"{BASE_URL}/api/chat", json=payload, headers=HEADERS)
        
        if response.status_code == 200:
            data = response.json()
            print(f"Actual Emotion: {data['emotion']}")
            print(f"Response Preview: {data['text'][:80]}...")
        else:
            print(f"Error: {response.status_code}")
        
        time.sleep(1)

def run_all_tests():
    """Run all test cases"""
    print_section("MENTAL HEALTH CHATBOT API TEST SUITE")
    print(f"Base URL: {BASE_URL}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    session_id = None
    
    try:
        # Test 1: Health Check
        results.append(("Health Check", test_health_check()))
        
        # Test 2: New Session
        session_id = test_chat_new_session()
        results.append(("Chat - New Session", session_id is not None))
        
        if session_id:
            # Test 3: Existing Session
            results.append(("Chat - Existing Session", test_chat_existing_session(session_id)))
            
            # Test 4: Multiple Messages
            test_multiple_messages(session_id)
            results.append(("Multiple Messages", True))
            
            # Test 5: Session Summary
            results.append(("Session Summary", test_session_summary(session_id)))
            
            # Test 6: Conversation History
            results.append(("Conversation History", test_conversation_history(session_id)))
        
        # Test 7: List Sessions
        results.append(("List Sessions", test_list_sessions()))
        
        # Test 8: Error Handling
        results.append(("Error Handling", test_invalid_request()))
        
        # Test 9: Delete Session
        if session_id:
            results.append(("Delete Session", test_delete_session(session_id)))
        
        # Test 10: Emotion Variety
        test_emotion_variety()
        results.append(("Emotion Variety", True))
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the API server.")
        print(f"   Please ensure the server is running at {BASE_URL}")
        print("   Start the server with: python api_server.py")
        return
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return
    
    # Print summary
    print_section("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed")
    print("\nDetails:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == "__main__":
    run_all_tests()
