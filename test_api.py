"""
Test script for Quran & Hadith QA API
Run after starting main.py server
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test 1: Check if API is running"""
    print("\n" + "="*70)
    print("TEST 1: Health Check")
    print("="*70)
    
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    return True

def test_single_question():
    """Test 2: Ask a single question"""
    print("\n" + "="*70)
    print("TEST 2: Single Question")
    print("="*70)
    
    question_data = {
        "question": "What is the importance of prayer in Islam?",
        "top_k": 3
    }
    
    try:
        response = requests.post(f"{BASE_URL}/ask", json=question_data)
        result = response.json()
        
        print(f"\n✅ Question: {result['question']}")
        print(f"📊 Confidence: {result['confidence']:.2%}")
        print(f"📚 Sources: {result['total_retrieved']}")
        print(f"\n📝 Answer Preview:")
        print(result['answer'][:300] + "...")
        
        print(f"\n📌 Top Sources:")
        for source in result['sources']:
            print(f"  • {source['type']}: {source['reference']}")
            print(f"    Score: {source['score']:.4f}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_multiple_questions():
    """Test 3: Ask different types of questions"""
    print("\n" + "="*70)
    print("TEST 3: Multiple Questions")
    print("="*70)
    
    questions = [
        "What does Islam teach about charity?",
        "How should Muslims treat their parents?",
        "What is fasting in Islam?"
    ]
    
    for i, q in enumerate(questions, 1):
        try:
            response = requests.post(
                f"{BASE_URL}/ask",
                json={"question": q, "top_k": 2}
            )
            result = response.json()
            
            print(f"\n[{i}] {q}")
            print(f"    Confidence: {result['confidence']:.2%}")
            print(f"    Sources: {result['total_retrieved']}")
            
        except Exception as e:
            print(f"❌ Error on question {i}: {e}")

def test_search():
    """Test 4: Simple search function"""
    print("\n" + "="*70)
    print("TEST 4: Search Function")
    print("="*70)
    
    try:
        response = requests.get(
            f"{BASE_URL}/search",
            params={"query": "prayer", "top_k": 3}
        )
        result = response.json()
        
        print(f"\n✅ Query: '{result['query']}'")
        print(f"📚 Found: {result['total_found']} results")
        
        for i, res in enumerate(result['results'][:3], 1):
            print(f"\n[{i}] Score: {res['score']:.4f}")
            print(f"    {res['source_type'].upper()}: {res['reference']}")
            print(f"    Text: {res['text'][:100]}...")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_stats():
    """Test 5: Get system statistics"""
    print("\n" + "="*70)
    print("TEST 5: System Statistics")
    print("="*70)
    
    try:
        response = requests.get(f"{BASE_URL}/stats")
        stats = response.json()
        
        print(f"\n📊 System Stats:")
        print(f"  Total Vectors: {stats['total_vectors']:,}")
        print(f"  Total Chunks: {stats['total_chunks']:,}")
        print(f"  Model: {stats['model_name']}")
        print(f"\n📚 Source Breakdown:")
        print(f"  Hadith: {stats['source_breakdown']['hadith']:,}")
        print(f"  Quran: {stats['source_breakdown']['quran']:,}")
        print(f"\n🌐 Languages: {', '.join(stats['languages'])}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING QURAN & HADITH QA API")
    print("="*70)
    print("\n⚠️  Make sure main.py is running in another terminal!")
    print("   Start it with: python main.py")
    
    input("\nPress Enter to start tests...")
    
    # Check if server is running
    if not test_health():
        print("\n❌ Server is not running!")
        print("Please start it first: python main.py")
        exit(1)
    
    # Run all tests
    try:
        test_single_question()
        test_multiple_questions()
        test_search()
        test_stats()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED!")
        print("="*70)
        print("\n💡 Try the interactive API docs at:")
        print("   http://localhost:8000/docs")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")