# -*- coding: utf-8 -*-
"""Created on Wed Feb 13 2019
@author: Jacob A. Smith
@version: 1.0

Extracts the environmental classification parameter from various output files
and matches the data to the master file.
"""
from astropy.table import QTable
import numpy as np

def build_ref_table( CROSS_REF_FILE_NAMES):
    """Compile the environmental classifications and identifying data from the
    various 'void_finder' output files.

    ATTN: Originally, the files in this project were cross referenced to Prof.
    Kelly Douglass' file via the link below. However, upon improvements to
    void_finder, new classifications wer given and the code has been updated.

    CROSS_REF_FILE_NAMES[0] .txt file obtained from:
    <http://www.pas.rochester.edu/~kdouglass/Research/
    kias1033_5_P-MJD-F_MPAJHU_Zdust_stellarMass_BPT_SFR_NSA_correctVflag.txt>

    @param:
        CROSS_REF_FILE_NAMES:
            string representations of the text files uses to extract the vflag
            parameter as obtained from void_finder

    @return:
        vflag_ref_table:
            astropy QTable containing the compiled vflag data from the output
            files

    """
    ###########################################################################
    # Initialize the 'vflag_ref_table' to contain the 'MaNGA_plate,'
    #    'MaNGA_fiberID,' and 'vflag' columns.
    #--------------------------------------------------------------------------
    vflag_ref_table = QTable( names=('MaNGA_plate', 'MaNGA_fiberID', 'vflag'),
                             dtype = ( int, int, int))
    ###########################################################################


    ###########################################################################
    # Read in each of the 'void_finder' output files.
    #--------------------------------------------------------------------------
    doug_not_classified = ascii.read( CROSS_REF_FILE_NAMES[1],
                           include_names = ('MaNGA_plate', 'MaNGA_fiberID',
                                            'vflag'), format='ecsv')
    doug_not_found = ascii.read( CROSS_REF_FILE_NAMES[2],
                           include_names = ('MaNGA_plate', 'MaNGA_fiberID',
                                            'vflag'), format='ecsv')
    doug_void_reclass = ascii.read( CROSS_REF_FILE_NAMES[3],
                           include_names = ('MaNGA_plate', 'MaNGA_fiberID',
                                            'vflag'), format='ecsv')
    doug_wall_reclass = ascii.read( CROSS_REF_FILE_NAMES[4],
                           include_names = ('MaNGA_plate', 'MaNGA_fiberID',
                                            'vflag'), format='ecsv')
    ###########################################################################


    ###########################################################################
    # For each row in each of the 'void_finder' output files, append that row
    #    onto the end of the 'vflag_ref_table.'
    #--------------------------------------------------------------------------
    for row in doug_not_classified:
        vflag_ref_table.add_row( row)

    for row in doug_not_found:
        vflag_ref_table.add_row( row)

    for row in doug_void_reclass:
        vflag_ref_table.add_row( row)

    for row in doug_wall_reclass:
        vflag_ref_table.add_row( row)
    ###########################################################################

    return vflag_ref_table

def match_vflag( master_table, vflag_ref_table, criteria):
    """Matches the vflag_ref_table to the master_table via MaNGA_plate and
    MaNGA_fiberID, then extracts the vflag parameter from the vflag_ref_table
    and assigns it to the corresponding entry in the master_table.

    @params:
        master_table:
            astropy QTable which the vflag_ref_table will be matched to

        vflag_ref_table:
            astropy QTable which the vflag parameter will be pulled from

        criteria:
            array of strings containing the column names which the data tables
            will be matched by

    @returns:
        master_table:
            updated version of master_table containing the extracted vflag
            parameter
    """
    master_table['vflag'] = -1 * np.ones( len( master_table))

    for i in range( len( master_table)):
        boolean = np.ones( len( vflag_ref_table), dtype=bool)

        for col_name in criteria:
            target_data = master_table[ col_name][i]

            bool_add = vflag_ref_table[ col_name] == target_data
            boolean = np.logical_and( boolean, bool_add)

        if sum( boolean) == 1:
            master_table['vflag'][i] = vflag_ref_table['vflag'][ boolean]
        elif sum( boolean) > 1:
            print("MULTIPLE MATCHES FOUND FOR GALAXY",
                  str( master_table['MaNGA_plate'][i]) + '-' \
                  + str( master_table['MaNGA_fiberID']))
        else:
            print("NO MATCHES FOUND FOR GALAXY",
                  str( master_table['MaNGA_plate'][i]) + '-' \
                  + str( master_table['MaNGA_fiberID']))

    return master_table