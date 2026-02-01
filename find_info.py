"""Find Model A info message in output."""
import sys

filepath = r'c:\Users\Comp\.cursor\projects\c-Users-Comp-GoldEA\agent-tools\7156b964-c8d6-4a15-84fe-facdccbb9b43.txt'

try:
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    keywords = ['[2/3]', 'Exit Mode', 'MODEL A', 'Account Balance', 'Running TREND']
    
    for i, line in enumerate(lines):
        for kw in keywords:
            if kw in line:
                start = max(0, i-2)
                end = min(len(lines), i+12)
                print(f'--- Found "{kw}" at line {i+1} ---')
                for j in range(start, end):
                    print(lines[j].rstrip()[:150])
                print()
                break
except Exception as e:
    print(f"Error: {e}")
