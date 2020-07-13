import glob
import ase.io
import numpy as np
from schnetpack import AtomsData
import pandas as pd
from ase.db import connect
# load the id to system and adsorbate lookups

adsorbate_mapping_csv = pd.read_csv('mapping_adsorbatesamplingindex_to_symbols.txt', ' ', header=None)

map_index_to_adsorbate = {}
for index, adsorbate in zip(adsorbate_mapping_csv[0], adsorbate_mapping_csv[1]):
    map_index_to_adsorbate[index]=adsorbate

bulk_mapping_csv = pd.read_csv('mapping_bulksamplingindex_to_mpid_and_symbols.txt', ' ', header=None)
map_index_to_bulk = {}
for index, mpid in zip(bulk_mapping_csv[0], bulk_mapping_csv[1]):
    map_index_to_bulk[index]=mpid

sysid_mapping_csv = pd.read_csv('sysid_to_bulk_adsorbate_samplingindex.txt', ' ', header=None)
map_sysid = {}
for sysid, bulk_index, adsorbate_index in zip(sysid_mapping_csv[0], sysid_mapping_csv[1],sysid_mapping_csv[2]):
    map_sysid[sysid]={'bulk':map_index_to_bulk[bulk_index],
                      'adsorbate':map_index_to_adsorbate[adsorbate_index]}

#Schnet doesn't like extra properties in the database like adsorbate, so we'll write a second one for that
with connect('ocp1k_adslab_train.db') as conn, connect('ocp1k_adslab_train_schnet.db') as conn_schnet:

    for trajfile in glob.glob('/tmp/train_traj_files/*.traj'):

        _atoms_list = ase.io.read(trajfile,'::10')

        for atoms in _atoms_list:
            # All properties need to be stored as numpy arrays.
            # Note: The shape for scalars should be (1,), not ()
            # Note: GPUs work best with float32 data
            energy = np.array([atoms.get_potential_energy()], dtype=np.float32)

            calc_properties=map_sysid[trajfile.split('/')[-1].split('_')[0]]
            properties= {'energy': energy, 
                         'forces': atoms.get_forces(apply_constraint=False)}

            conn.write(atoms, data=properties, **calc_properties)
            conn_schnet.write(atoms, data=properties)


with connect('ocp1k_adslab_val.db') as conn, connect('ocp1k_adslab_val_schnet.db') as conn_schnet:

    for trajfile in glob.glob('/tmp/val_traj_files/*.traj'):

        _atoms_list = ase.io.read(trajfile,'::10')

        for atoms in _atoms_list:
            # All properties need to be stored as numpy arrays.
            # Note: The shape for scalars should be (1,), not ()
            # Note: GPUs work best with float32 data
            energy = np.array([atoms.get_potential_energy()], dtype=np.float32)

            calc_properties=map_sysid[trajfile.split('/')[-1].split('_')[0]]
            properties= {'energy': energy,
                 'forces': atoms.get_forces(apply_constraint=False)}

            conn.write(atoms, data=properties, **calc_properties)
            conn_schnet.write(atoms, data=properties)

