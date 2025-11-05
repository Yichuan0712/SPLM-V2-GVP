import math
import h5py


def load_h5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        seq = f['seq'][()]
        n_ca_c_o_coord = f['N_CA_C_O_coord'][:]
        plddt_scores = f['plddt_scores'][:]
    return seq, n_ca_c_o_coord, plddt_scores


# seq, n_ca_c_o_coord, plddt_scores,ss,sasa,node_degree,edge_index,edge_dis,edge_localorientation=load_h5_file_v2("/data/duolin/train_PLK1/datasets/structuresAs_hdf5_v2/AF-Q8WWH5-F1-model_v4_chain_id_A.h5")
def load_h5_file_v2(file_path):
    with h5py.File(file_path, 'r') as f:
        seq = f['seq'][()]
        n_ca_c_o_coord = f['N_CA_C_O_coord'][:]
        plddt_scores = f['plddt_scores'][:]
        ss = f['ss'][:]
        ss = ''.join([s.decode('utf-8') for s in ss])
        sasa = f['sasa'][:]
        if "HB_energy" in f:
            HB_energy, HB_offset, HB_degree = f['HB_energy'][:], f['HB_offset'][:], f['HB_degree'][:]
        else:
            HB_energy, HB_offset, HB_degree = None, None, None
        node_degree = f['node_degree'][:]
        edge_index = f['edge_index'][:]
        edge_dis = f['edge_dis'][:]
        edge_localorientation = f['edge_localorientation'][:]

    return seq, n_ca_c_o_coord, plddt_scores, ss, sasa, HB_energy, HB_offset, HB_degree, node_degree, edge_index, edge_dis, edge_localorientation


def write_h5_file(file_path, pad_seq, n_ca_c_o_coord, plddt_scores):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('seq', data=pad_seq)
        f.create_dataset('N_CA_C_O_coord', data=n_ca_c_o_coord)
        f.create_dataset('plddt_scores', data=plddt_scores)


def extract_coordinates_plddt(chain, max_len):
    """Returns a list of C-alpha coordinates for a chain"""
    pos = []
    plddt_scores = []
    for row, residue in enumerate(chain):
        pos_N_CA_C_O = []
        plddt_scores.append(residue["CA"].get_bfactor())
        for key in ['N', 'CA', 'C', 'O']:
            if key in residue:
                pos_N_CA_C_O.append(list(residue[key].coord))
            else:
                pos_N_CA_C_O.append([math.nan, math.nan, math.nan])

        pos.append(pos_N_CA_C_O)
        if len(pos) == max_len:
            break

    return pos, plddt_scores


if __name__ == '__main__':
    print('for test')
    prot_seq, coords, scores = load_h5_file('./save_test/AF-G5EB01-F1-model_v4.h5')
    print(prot_seq, coords)
