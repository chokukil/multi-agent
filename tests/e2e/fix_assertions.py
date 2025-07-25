#!/usr/bin/env python3
"""
Script to add descriptive messages to all assert statements in E2E tests
Addresses QA analysis finding: 37 assertions without descriptive messages
"""

import re
import os
from pathlib import Path

def fix_assertions_in_file(file_path: Path) -> int:
    """Fix assertions in a single file"""
    if not file_path.exists():
        return 0
        
    content = file_path.read_text()
    original_content = content
    fixes_count = 0
    
    # Pattern to match assert statements without messages
    # Looks for: assert condition, assert condition == value, etc.
    # Excludes: assert condition, "message"
    assert_pattern = r'assert\s+([^,\n]+?)(?:\s*,\s*["\']([^"\']*)["\'])?\s*$'
    
    lines = content.split('\n')
    new_lines = []
    
    for line_num, line in enumerate(lines, 1):
        # Skip lines that already have assertion messages
        if 'assert' in line and ',' in line and ('"' in line or "'" in line):
            new_lines.append(line)
            continue
        
        # Look for assert statements without messages
        match = re.search(r'^\s*assert\s+([^,\n]+?)(?:\s*#.*)?$', line)
        if match:
            indent = re.match(r'^(\s*)', line).group(1)
            condition = match.group(1).strip()
            
            # Generate meaningful message based on condition
            message = generate_assertion_message(condition, file_path.name, line_num)
            
            # Add message to assertion
            new_line = f'{indent}assert {condition}, "{message}"'
            new_lines.append(new_line)
            fixes_count += 1
        else:
            new_lines.append(line)
    
    if fixes_count > 0:
        file_path.write_text('\n'.join(new_lines))
        print(f"Fixed {fixes_count} assertions in {file_path.name}")
    
    return fixes_count

def generate_assertion_message(condition: str, filename: str, line_num: int) -> str:
    """Generate a meaningful assertion message based on the condition"""
    
    # Clean up condition for analysis
    condition = condition.strip()
    
    # Common patterns and their messages
    if 'health_status.get(' in condition:
        agent_name = extract_agent_name(condition)
        return f"{agent_name} agent must be healthy for this test"
    
    elif '.is_visible()' in condition:
        element = extract_element_name(condition)
        return f"{element} should be visible"
    
    elif '.wait_for_' in condition:
        operation = extract_wait_operation(condition)
        return f"{operation} should complete successfully"
    
    elif 'len(' in condition and '>=' in condition:
        return "Should have sufficient number of items"
    
    elif 'len(' in condition and '>' in condition:
        return "Should have at least one item"
    
    elif 'in ' in condition and '.lower()' in condition:
        search_term = extract_search_term(condition)
        return f"Response should contain '{search_term}'"
    
    elif condition.startswith('await '):
        operation = condition.replace('await ', '').split('(')[0]
        return f"{operation} operation should succeed"
    
    elif '==' in condition:
        expected = condition.split('==')[1].strip()
        return f"Value should equal {expected}"
    
    elif '!=' in condition:
        not_expected = condition.split('!=')[1].strip()
        return f"Value should not equal {not_expected}"
    
    elif '.status_code ==' in condition:
        return "HTTP request should return expected status code"
    
    elif 'response' in condition.lower():
        return "Response should meet expected criteria"
    
    elif 'upload' in condition.lower():
        return "File upload should complete successfully"
    
    elif 'agent' in condition.lower():
        return "Agent operation should complete as expected"
    
    elif 'chart' in condition.lower() or 'visualization' in condition.lower():
        return "Chart/visualization should render correctly"
    
    else:
        # Generic message with context
        return f"Condition should be true: {condition[:50]}{'...' if len(condition) > 50 else ''}"

def extract_agent_name(condition: str) -> str:
    """Extract agent name from health check condition"""
    match = re.search(r'["\'](.*?)["\']', condition)
    return match.group(1).replace('_', ' ').title() if match else "Agent"

def extract_element_name(condition: str) -> str:
    """Extract element description from visibility condition"""
    if 'upload' in condition.lower():
        return "Upload element"
    elif 'button' in condition.lower():
        return "Button"
    elif 'message' in condition.lower():
        return "Message"
    else:
        return "Element"

def extract_wait_operation(condition: str) -> str:
    """Extract operation name from wait condition"""
    if 'wait_for_file_upload' in condition:
        return "File upload"
    elif 'wait_for_agent_response' in condition:
        return "Agent response"
    elif 'wait_for_analysis' in condition:
        return "Analysis"
    elif 'wait_for_chart' in condition:
        return "Chart rendering"
    else:
        return "Operation"

def extract_search_term(condition: str) -> str:
    """Extract search term from string search condition"""
    # Look for patterns like: "keyword" in response.lower()
    matches = re.findall(r'["\'](.*?)["\']', condition)
    return matches[0] if matches else "expected content"

def main():
    """Fix assertions in all E2E test files"""
    test_dir = Path(__file__).parent
    
    test_files = [
        "test_user_journeys.py",
        "test_agent_collaboration.py", 
        "test_error_recovery.py",
        "test_agent_collaboration_fixed.py"
    ]
    
    total_fixes = 0
    
    for filename in test_files:
        file_path = test_dir / filename
        if file_path.exists():
            fixes = fix_assertions_in_file(file_path)
            total_fixes += fixes
        else:
            print(f"File not found: {filename}")
    
    print(f"\nTotal assertions fixed: {total_fixes}")
    print("All E2E test files have been updated with descriptive assertion messages!")

if __name__ == "__main__":
    main()