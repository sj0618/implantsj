import re
import sys
from pathlib import Path

import pandas as pd


OPEN = {
    'simpleline', 'exfeel internal', 'ss iii', 'biohorizon external', 'tsv',
    's clean tapered', 'gs iii', 'screw vent', 'atlas internal', 'stella',
    '3i', 'ankylos', 'iu', 'replace select', 'core1',
}


def key(value):
    value = re.sub(r'^\s*\d+\s*\.\s*', '', str(value))
    return re.sub(r'\s*\([^()]*\)\s*$', '', value).strip().casefold()


assert len(sys.argv) == 4, 'usage: script FULL_CSV OPEN_OUT CLOSED_OUT'
source, open_out, closed_out = map(Path, sys.argv[1:])
assert len({source.resolve(), open_out.resolve(), closed_out.resolve()}) == 3
df = pd.read_csv(source)
assert {'label', 'split'} <= set(df.columns), 'manifest needs label,split'
keys = df['label'].map(key)
assert len(set(keys)) == 42, 'normalized labels must be unique'
assert not OPEN - set(keys), f'missing open labels: {sorted(OPEN - set(keys))}'
mask = keys.isin(OPEN)
groups = [('open', df[mask].copy(), open_out, 15),
          ('closed', df[~mask].copy(), closed_out, 27)]
raw = set(df['label'].astype(str))
labels = [set(part['label'].astype(str)) for _, part, _, _ in groups]
assert len(raw) == 42 and len(labels[0]) == 15 and len(labels[1]) == 27
assert labels[0].isdisjoint(labels[1]) and labels[0] | labels[1] == raw
assert sum(len(part) for _, part, _, _ in groups) == len(df)
assert all(set(part['split'].astype(str)) >= {'train', 'valid', 'test'}
           for _, part, _, _ in groups)
for name, part, out, count in groups:
    out.parent.mkdir(parents=True, exist_ok=True)
    part.to_csv(out, index=False)
    splits = part['split'].astype(str).value_counts().sort_index().to_dict()
    print(f'{name}: rows={len(part)} labels={count} splits={splits} out={out}')
