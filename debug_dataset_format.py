import webdataset as wds
import json
from pathlib import Path

dataset_path = Path('data/elise_webdataset')
shards = sorted(list(dataset_path.glob('*.tar')))
dataset = wds.WebDataset(str(shards[0]))
sample = next(iter(dataset))

print("=" * 60)
print("Dataset Sample Structure")
print("=" * 60)

print("\nTop-level keys:")
for key in sample.keys():
    print(f"  - {key}: {type(sample[key])}")

print("\nJSON content:")
sample_data = json.loads(sample['json'])
for key in sample_data.keys():
    value = sample_data[key]
    if isinstance(value, list):
        print(f"  - {key}: list with {len(value)} items")
        if len(value) > 0:
            print(f"    First item type: {type(value[0])}")
    else:
        print(f"  - {key}: {type(value).__name__}")
        if not isinstance(value, (list, dict)):
            print(f"    Value: {value}")

print("\n" + "=" * 60)
