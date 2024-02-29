"""

"""

import numpy as np


def main():
    listo = []
    ref_list = [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],[0, 1, 2, 3, 4, 5, 6, 7, 8, 1]]]
    for i in ref_list:
        for j in i:
            vec = j.copy()
            vec.sort()
            listo.append(vec)
    print(f'listo: {listo}')
    all_faces = np.array(listo)
    print(f'all_faces: {all_faces}')
    unique_faces = np.unique(all_faces, axis=0)
    print(f'unique_faces: {unique_faces}')

if __name__ == '__main__':
    main()
