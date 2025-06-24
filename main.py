import os
from enum import Enum

from extract_frames import extract_all_test_frames


class DishesType(Enum):
    tea = 1
    greek_salat = 2
    ribs = 3
    pita = 4
    salat_2 = 5
    borsch = 6
    pumpkin_soup = 7
    vodka = 8
    pickled_onions = 9


if __name__ == '__main__':
    extract_all_test_frames()
