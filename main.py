import os

from pathlib import Path


class_id = {
    0: 'Tea',
    1: 'greek_salat',
    2: 'ribs',
    3: 'pita',
    4: 'salat_2',
    5: 'borsch',
    6: 'pumpkin_soup',
    7: 'vodka',
    8: 'pickled_onions',
}

ROOT_DIR = Path(os.getcwd())
RUNS_DIR = ROOT_DIR / 'runs'
