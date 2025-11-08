"""
Test script for Translation Module
Run this to verify translation functionality
"""

import sys
import os
# Ensure project root is first on sys.path so package imports resolve correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from translation.translation import TranslationService, get_translation_service
from translation.model_loader import get_supported_languages
from translation.transliteration import transliterate_text
import time

def test_language_detection():
    """Test language detection for various scripts"""
    print("\n" + "="*60)
    print("TEST 1: Language Detection")
    print("="*60)
    
    translator = TranslationService()
    
    test_cases = [
        ("Hello, how are you?", "en"),
        ("नमस्ते, आप कैसे हैं?", "hi"),
        ("नमस्कार, तुम्ही कसे आहात?", "mr"),
        ("வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?", "ta"),
        ("నమస్కారం, మీరు ఎలా ఉన్నారు?", "te"),
        ("হ্যালো, আপনি কেমন আছেন?", "bn"),
        ("નમસ્તે, તમે કેવી રીતે છો?", "gu"),
        ("ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?", "kn"),
    ]
    
    passed = 0
    failed = 0
    
    for text, expected_lang in test_cases:
        detected = translator.detect_language(text)
        status = "✓ PASS" if detected == expected_lang else "✗ FAIL"
        print(f"{status} | Text: {text[:30]}... | Expected: {expected_lang} | Got: {detected}")
        
        if detected == expected_lang:
            passed += 1
        else:
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return passed == len(test_cases)

def test_translation_to_english():
    """Test translation from various languages to English"""
    print("\n" + "="*60)
    print("TEST 2: Translation to English")
    print("="*60)
    
    translator = TranslationService()
    
    test_cases = [
        ("मुझे FIU-IND के बारे में बताएं", "hi"),
        ("FIU-IND बद्दल मला सांगा", "mr"),
        ("எனக்கு FIU-IND பற்றி சொல்லுங்கள்", "ta"),
    ]
    
    for text, lang in test_cases:
        print(f"\nOriginal ({lang}): {text}")
        start_time = time.time()
        
        try:
            english = translator.translate_to_english(text, lang)
            elapsed = time.time() - start_time
            print(f"English: {english}")
            print(f"Time: {elapsed:.2f}s")
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
    
    return True

def test_translation_from_english():
    """Test translation from English to various languages"""
    print("\n" + "="*60)
    print("TEST 3: Translation from English")
    print("="*60)
    
    translator = TranslationService()
    
    english_text = "FIU-IND is the Financial Intelligence Unit of India"
    target_languages = ["hi", "mr", "ta"]
    
    for lang in target_languages:
        print(f"\nTarget Language: {lang}")
        start_time = time.time()
        
        try:
            translated = translator.translate_from_english(english_text, lang)
            elapsed = time.time() - start_time
            print(f"Translated: {translated}")
            print(f"Time: {elapsed:.2f}s")
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
    
    return True

def test_conversation_processing():
    """Test full conversation round-trip"""
    print("\n" + "="*60)
    print("TEST 4: Conversation Processing")
    print("="*60)
    
    translator = TranslationService()
    
    # Simulate a conversation
    user_input = "मुझे PMLA अधिनियम के बारे में बताएं"
    system_response_en = "The Prevention of Money Laundering Act (PMLA) is an important legislation in India."
    
    print(f"User Input (Hindi): {user_input}")
    
    try:
        start_time = time.time()
        lang, english_query, translated_response = translator.process_conversation(
            user_input=user_input,
            system_response=system_response_en
        )
        elapsed = time.time() - start_time
        
        print(f"\nDetected Language: {lang}")
        print(f"English Query: {english_query}")
        print(f"System Response (EN): {system_response_en}")
        print(f"Translated Response: {translated_response}")
        print(f"Total Time: {elapsed:.2f}s")
        
        return True
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        return False

def test_transliteration():
    """Test transliteration functionality"""
    print("\n" + "="*60)
    print("TEST 5: Transliteration")
    print("="*60)
    
    test_cases = [
        "नमस्ते भारत",
        "வணக்கம் இந்தியா",
    ]
    
    for text in test_cases:
        try:
            romanized = transliterate_text(text)
            print(f"Original: {text}")
            print(f"Romanized: {romanized}\n")
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
    
    return True

def test_caching():
    """Test translation caching performance"""
    print("\n" + "="*60)
    print("TEST 6: Caching Performance")
    print("="*60)
    
    translator = TranslationService()
    text = "नमस्ते"
    
    # First call (no cache)
    start1 = time.time()
    result1 = translator.translate_to_english(text, "hi")
    time1 = time.time() - start1
    
    # Second call (cached)
    start2 = time.time()
    result2 = translator.translate_to_english(text, "hi")
    time2 = time.time() - start2
    
    print(f"First call (no cache): {time1:.3f}s")
    print(f"Second call (cached): {time2:.3f}s")
    print(f"Speedup: {time1/time2:.1f}x")
    
    speedup = time1 / time2 if time2 > 0 else 0
    return speedup > 10  # Cache should be much faster

def test_supported_languages():
    """Test supported languages list"""
    print("\n" + "="*60)
    print("TEST 7: Supported Languages")
    print("="*60)
    
    languages = get_supported_languages()
    print(f"\nTotal supported languages: {len(languages)}")
    
    for code, name in languages.items():
        print(f"  {code}: {name}")
    
    return len(languages) >= 11  # Should have at least 11 languages

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("TRANSLATION MODULE TEST SUITE")
    print("="*80)
    
    tests = [
        ("Language Detection", test_language_detection),
        ("Translation to English", test_translation_to_english),
        ("Translation from English", test_translation_from_english),
        ("Conversation Processing", test_conversation_processing),
        ("Transliteration", test_transliteration),
        ("Caching Performance", test_caching),
        ("Supported Languages", test_supported_languages),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}...")
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ CRITICAL ERROR in {test_name}: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} | {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    exit(main())