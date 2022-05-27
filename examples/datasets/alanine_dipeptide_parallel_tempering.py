import os
import torch
import numpy as np
from tqdm import tqdm
import urllib.request as request

USER_AGENT = "pytorch/vision"

angles_file = "https://ftp.mi.fu-berlin.de/pub/cmb-data/alanine_dipeptide_parallel_tempering_dihedrals.npz"
energies_file = "https://ftp.mi.fu-berlin.de/pub/cmb-data/alanine_dipeptide_parallel_tempering_energies.npz"


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with request.urlopen(request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)

def get_abs_path(filename):
    dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(dir, filename)


def _remove_if_exists(filename):
    if os.path.exists(filename):
        os.remove(filename)


def _download_dataset():
    filename = get_abs_path('angles.npz')
    _remove_if_exists(filename)
    _urlretrieve(url=angles_file, filename=filename)

    filename = get_abs_path('energies.npz')
    _remove_if_exists(filename)
    _urlretrieve(url=energies_file, filename=filename)


def get_data():
    _download_dataset()

    energies = []
    trajs = []

    temperatures = torch.arange(300, 501, 10)
    with open(get_abs_path('angles.npz')):
        angles_data = np.load(get_abs_path('angles.npz'))
    with open(get_abs_path('energies.npz')):
        energies_data = np.load(get_abs_path('energies.npz'))

    for t in temperatures:
        trajs.append(angles_data[f't{t}'])
        energies.append(energies_data[f't{t}'])

    return trajs, energies