"""
Quick test script to verify enhanced detector is working
"""

print("Testing Enhanced Detector Installation...")
print("=" * 50)

# Test 1: Check imports
print("\n1. Checking imports...")
try:
    import transformers
    import torch
    print(f"   ✅ transformers {transformers.__version__}")
    print(f"   ✅ torch {torch.__version__}")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    print("   Run: pip install transformers torch sentencepiece")
    exit(1)

# Test 2: Test enhanced detector
print("\n2. Testing Enhanced Detector...")
try:
    from enhanced_detector import EnhancedAIDetector
    print("   ✅ Enhanced detector module loaded")
    
    # Initialize detector
    print("   Initializing transformer model (this may take a minute on first run)...")
    detector = EnhancedAIDetector(method="transformer")
    print("   ✅ Transformer detector initialized")
    
    # Test prediction
    test_text = "The implementation of this system demonstrates a comprehensive approach to data analysis."
    print(f"\n3. Testing prediction on sample text...")
    print(f"   Text: '{test_text[:50]}...'")
    
    result = detector.predict(test_text)
    print(f"\n   Results:")
    print(f"   - Method: {result.get('method', 'unknown')}")
    print(f"   - AI Probability: {result.get('ai_probability', 0):.2%}")
    print(f"   - Human Probability: {result.get('human_probability', 0):.2%}")
    print(f"   - Confidence: {result.get('confidence', 0):.2%}")
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("Enhanced detector is ready to use in your app.")
    print("\nRun: streamlit run app.py")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)



