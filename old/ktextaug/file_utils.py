from pathlib import Path

def open_text(fn, enc='utf-8'):
    with open(fn,'r', encoding = enc) as f: return ''.join(f.readlines())

def save_texts(fname, texts):
    with open(fname, 'w') as f:
        for t in texts:
            f.write(f'{t}\n')


def make_dir_structure(path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)