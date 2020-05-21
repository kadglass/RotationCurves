
from astropy.table import QTable, Table, vstack
from astropy.io import fits




################################################################################
# Import galaxy data
#-------------------------------------------------------------------------------
galaxies_filename = 'Pipe3D-master_file_vflag_10_smooth.txt'

galaxies = QTable.read(galaxies_filename, format='ascii.ecsv')
################################################################################



################################################################################
# Filtering tables
#-------------------------------------------------------------------------------
remove_more_points_table = Table.read('../Not_enough_points_removed.txt', format='ascii.commented_header')
no_rotation_table = Table.read('../No_rotation.txt', format='ascii.commented_header')
interacting_table = Table.read('../Interacting.txt', format='ascii.commented_header')
bad_center_table = Table.read('../Bad_center.txt', format='ascii.commented_header')
bad_phi_table = Table.read('../Bad_angle.txt', format='ascii.commented_header')
QSO_table = Table.read('../QSO.txt', format='ascii.commented_header')

bad_galaxies = vstack([no_rotation_table, 
                       remove_more_points_table, 
                       bad_phi_table, 
                       interacting_table, 
                       bad_center_table, 
                       QSO_table])
#-------------------------------------------------------------------------------
# Build dictionary to hold plate, fiber combinations of all additional bad 
# galaxies
#-------------------------------------------------------------------------------
bad_galaxies_dict = {}

for i in range(len(bad_galaxies)):

    ID = (bad_galaxies['MaNGA_plate'][i], bad_galaxies['MaNGA_fiberID'][i])

    bad_galaxies_dict[ID] = 1
################################################################################



################################################################################
# Filter galaxies
#-------------------------------------------------------------------------------
for i in range(len(galaxies)):

    if galaxies['curve_used'][:3] is not 'non':

        plate = galaxies['MaNGA_plate'][i]
        IFU = galaxies['MaNGA_IFU'][i]

        galaxy_filename = '../data/MaNGA/MaNGA_DR15/pipe3d/' + str(plate) + '/manga-' + str(plate) + '-' + str(IFU) + '.Pipe3D.cube.fits.gz'

        galaxy_file = fits.open(galaxy_filename)

        org_hdr = galaxy_file[0].header

        galaxy_file.close()




        MaNGA_galaxy_target = org_hdr['MNGTARG1']
        DRP_3D_quality = org_hdr['DRP3QUAL']

        map_smoothness = galaxies['smoothness_score'][i]

        if (MaNGA_galaxy_target == 0) or \
           (DRP_3D_quality > 10000) or \
           (map_smoothness > 2.27) or \
           ((plate, IFU) in bad_galaxies_dict.keys()):
            galaxies['curve_used'][i] = 'non'
################################################################################



################################################################################
# Save results
#-------------------------------------------------------------------------------
galaxies.write(galaxies_filename[:-4] + '2p27.txt', format='ascii.ecsv', overwrite=True)
################################################################################


