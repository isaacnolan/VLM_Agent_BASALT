#!/usr/bin/env python3
"""
Test JSON parsing with the improved error handling.
"""
import json
import re

def clean_and_parse_json(raw_text):
    """Test the cleaning logic."""
    # Extract JSON
    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON found in response")
    
    raw_json = json_match.group()
    print("=" * 80)
    print("RAW JSON:")
    print(raw_json)
    print("=" * 80)
    
    # Clean up
    clean = raw_json
    clean = re.sub(r'(?P<key>"[a-zA-Z0-9_.]+"\s*:\s*)(?:0\s*or\s*1|1\s*or\s*0)', r"\g<key>1", clean)
    clean = re.sub(r"\bTrue\b", "true", clean)
    clean = re.sub(r"\bFalse\b", "false", clean)
    clean = clean.replace("\'", '"')
    # Remove trailing commas
    clean = re.sub(r",\s*(\}|\])", r"\1", clean)
    # Remove duplicate commas
    clean = re.sub(r",\s*,", ",", clean)
    # Fix missing commas between properties
    clean = re.sub(r'([0-9\]\}])\s*\n\s*"', r'\1,\n"', clean)
    # Remove comments
    clean = re.sub(r'//.*$', '', clean, flags=re.MULTILINE)
    
    print("\nCLEANED JSON:")
    print(clean)
    print("=" * 80)
    
    try:
        result = json.loads(clean)
        print("\nPARSED SUCCESSFULLY!")
        print(json.dumps(result, indent=2))
        return result
    except json.JSONDecodeError as e:
        print(f"\nJSON ERROR: {e}")
        print(f"At position {e.pos}: ...{clean[max(0,e.pos-50):e.pos+50]}...")
        
        # Try to extract just actions array
        actions_match = re.search(r'"actions"\s*:\s*\[(.*?)\]', clean, re.DOTALL)
        if actions_match:
            print("\nTrying to extract just the actions array...")
            actions_str = "[" + actions_match.group(1) + "]"
            try:
                actions_array = json.loads(actions_str)
                result = {"actions": actions_array, "reasoning": "Partially recovered"}
                print("RECOVERED!")
                print(json.dumps(result, indent=2))
                return result
            except:
                pass
        
        raise


# Test case 1: Valid JSON
test1 = """
{
  "reasoning": "Test",
  "actions": [
    {"step_reasoning": "Move", "action": {"forward": 1, "camera": [0, 0]}}
  ]
}
"""

# Test case 2: Missing commas (common VLM error)
test2 = """
{
  "reasoning": "Test"
  "actions": [
    {"step_reasoning": "Move" "action": {"forward": 1}}
  ]
}
"""

# Test case 3: Trailing commas
test3 = """
{
  "reasoning": "Test",
  "actions": [
    {"step_reasoning": "Move", "action": {"forward": 1,}},
  ],
}
"""

print("TEST 1: Valid JSON")
try:
    clean_and_parse_json(test1)
except Exception as e:
    print(f"FAILED: {e}")

print("\n\nTEST 2: Missing commas")
try:
    clean_and_parse_json(test2)
except Exception as e:
    print(f"FAILED: {e}")

print("\n\nTEST 3: Trailing commas")
try:
    clean_and_parse_json(test3)
except Exception as e:
    print(f"FAILED: {e}")
