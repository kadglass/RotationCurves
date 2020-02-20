



def galaxies_dict(ref_table):
    '''
    Build a dictionary of the galaxies with NSA IDs

    Parameters:
    ===========

    ref_table : astropy table
        Galaxies with NSA IDs


    Returns:
    ========

    ref_dict : dictionary
        Keys are the NSA ID, value is the index number in the NSA catalog
    '''

    # Initialize dictionary of cell IDs with at least one galaxy in them
    ref_dict = {}

    for idx in range(len(ref_table)):

        galaxy_ID = (ref_table['NSAID'][idx])

        ref_dict[galaxy_ID] = idx

    return ref_dict