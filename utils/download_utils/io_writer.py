import awkward as ak

def _write_outputs_to_root(path, rootdir_dict, compression=-1, step=1048576):
    '''
    Adapted from Weaver
    Now supports multi-tree writing
    rootdir_dict must be an Awkward Array in the form {'TTree': {'branch_name': __}}
    '''
    import uproot
    if compression == -1:
        compression = uproot.LZ4(4)
    
    with uproot.recreate(path, compression=compression) as fout:

        print(f'Writing outputs to ROOT file at: {path}')
        
        for t_tree, branch in rootdir_dict.items():

            table = ak.Array(branch)
            name = t_tree

            print(f'Processing TTree: {name}')
            
            tree = fout.mktree(name, {k: table[k].type for k in table.fields})
            start = 0
            while start < len(table[table.fields[0]]) - 1:
                tree.extend({k: table[k][start:start + step] for k in table.fields})
                start += step

def _hook_output_handler(outputs, library = 'ak'):
    '''
    Accepts detached tensors on gpu
    Converts per-batch tensor outputs to awkward
    Currently only supports logits
    '''

    import torch
    cpu_outputs = torch.cat(outputs, dim = 0).cpu()

    if library == 'ak':
        return ak.Array(cpu_outputs.numpy())
    if library == 'np':
        return cpu_outputs.numpy()
    else:
        print(f"Library {library} not supported. Using basic Python list")
        return cpu_outputs.tolist()