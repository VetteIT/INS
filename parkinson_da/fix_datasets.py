"""Removes duplicate old code from datasets.py"""
with open('src/data/datasets.py', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

print(f'Total lines before: {len(lines)}')
idx = None
count = 0
for i, l in enumerate(lines):
    if 'class ParkinsonDataset' in l:
        count += 1
        if count == 2:
            idx = i
            break

if idx:
    print(f'Removing duplicate from line {idx+1}')
    with open('src/data/datasets.py', 'w', encoding='utf-8') as f:
        f.writelines(lines[:idx])
    print(f'Done. Remaining lines: {idx}')
else:
    print('No duplicate found.')
