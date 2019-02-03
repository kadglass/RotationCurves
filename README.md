# RotationCurves
Measuring the rotation curves of SDSS MaNGA galaxies.

‘rotation_curve_vX_X’ is configured to receive FITS files from the SDSS MaNGA Pipe3D catalog. MaNGA data can be downloaded from the [DR14 MaNGA database]( https://dr14.sdss.org/sas/dr14/manga/spectro/pipe3d/v2_1_2/2.1.2/) (see [instructions]( http://www.sdss.org/dr14/manga/manga-data/data-access/)) and [DR15 MaNGA database]( https://dr15.sdss.org/sas/dr15/manga/spectro/pipe3d/v2_4_3/2.4.3/) (see [instructions]( http://www.sdss.org/dr15/manga/manga-data/data-access/)). All MaNGA data follows the [MaNGA datamodel](https://data.sdss.org/datamodel/files/MANGA_PIPE3D/MANGADRP_VER/PIPE3D_VER/PLATE/manga.Pipe3D.cube.html). In addition, the NASA-Sloan-Atlas (NSA) catalog, [nsa_v0_1_2](http://sdss.physics.nyu.edu/mblanton/v0/nsa_v0_1_2.fits), is used to cross reference for stellar mass. All data contained within the NSA catalog follows the NSA datamodel found at the [NSA website](http://nsatlas.org/data).

The output of ‘rotation_curve_vX_X’ is two ‘.txt’ files in ‘ecsv’ format for each MaNGA data file housed in ‘/rot_curve_data_files’. 
The first ‘.txt’ file is of the format “dr[NUMBER OF DATA RELEASE]-[MANGA PLATE]-[MANGA FIBER ID]_rot_curve_data” and contains the following quantities as a function of deprojected radius:
maximum velocity in units of km /s
error in maximum velocity in units of km /s
minimum velocity in units of km /s
error in minimum velocity in units of km /s
average (between the maximum and minimum) velocity in units of km /s
error in the average velocity in units of km /s
difference between the maximum and minimum velocity in units of km /s
error in the difference between the maximum and minimum velocity in units of km /s
total mass interior in units of solar masses
error in total mass interior in units of solar masses
stellar mass interior in units of solar masses
stellar component of the rotational velocity in units of km /s
error in the stellar component of the rotational velocity in units of km /s
dark matter mass interior in units of solar masses
error in dark matter mass interior in units of solar masses
dark matter component of the rotational velocity in units of km /s
error in the dark matter component of the rotational velocity in units of km /s.

The second ‘.txt’ file is of the format “dr[NUMBER OF DATA RELEASE]-[MANGA PLATE]-[MANGA FIBER ID]_gal_stat_data” and contains a string identifier of the format ““dr[NUMBER OF DATA RELEASE]-[MANGA PLATE]-[MANGA FIBER ID],” the luminosity of the brightest spaxel in the visual band, and its error both in units of solar luminosity.
