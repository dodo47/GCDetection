# Explanation of tabular features in ACS_sources_original.csv

See also photutil's segmentation routines: https://photutils.readthedocs.io/en/stable/segmentation.html \
And photutil's aperture photometry routines: https://photutils.readthedocs.io/en/stable/aperture.html

Columns:
 -   `ID:`                  ID in the individual galaxy
 -   `galaxy:`              galaxy name  
 -   `label:`               ID label from the source extraction
 -   `xcentroid:`           x-pixel of source centroid in the g band filter (F475W filter)
 -   `ycentroid:`           y-pixel of source centroid in the g band filter (F850LP filter)
 -   `sky_centroid.ra:`     centroid in right ascension in g band
 -   `sky_centroid.dec:`    centroid in declination in g band
 -   `area:`                number of pixels above detection threshold in g band
 -   `semimajor_sigma:`     extension along semimajor axis in g band
 -   `semiminor_sigma:`     extension along semiminor axis in g band
 -  `orientation:`         angular orientation on image in g band
 -  `eccentricity:`        eccentricity of source in g band
 -  `min_value:`           minimum value of all pixels in source in g band
 -  `max_value:`           max value in g band
 -  `segment_flux:`        integrated flux of all the pixels in the source in g band
 -  `kron_flux:`           some other flux measurement in g band
 -  `label_z:`             see above  
 -  `xcentroid_z:`         "
 -  `ycentroid_z:`         "
 -  `sky_centroid.ra_z:`   "
 -  `sky_centroid.dec_z:`  "
 -  `area_z:`              "
 -  `semimajor_sigma_z:`   "
 -  `semiminor_sigma_z:`   "
 -  `orientation_z:`       "
 -  `eccentricity_z:`      "
 -  `min_value_z:`         "
 -  `max_value_z:`         "
 -  `segment_flux_z:`      "
 -  `kron_flux_z:`         "
 -  `HST_ID:`              ID of the source in the ACSFCS catalogue (only if matched, otherwise nan)
 -  `pGC:`                 probabilty that a source is a GC, from ACSFCS catalogue (0 if not matched)
 -  `matched:`             bool whether it is matched with the ACSFCS catalogue or not
 -  `CI3_g:`               concentration index measurement (measurement of magnitude between 3 pix aperture and 1 pix) in g band
 -  `CI3_z:`               in z
 -  `m3_g:`                some simple measure of magnitude (brightness) in a 3 pixel aperture in g band
 -  `m3_z:`                in z
 -  `CI4_g:`               concentration index measurement (measurement of magnitude between 4 pix aperture and 1 pix) in g band
 -  `CI4_z:`               in z
 -  `m4_g:`                some simple measure of magnitude (brightness) in a 4 pixel aperture in g band
 -  `m4_z:`                in z
 -  `CI5_g:`               concentration index measurement (measurement of magnitude between 5 pix aperture and 1 pix) in g band
 -  `CI5_z:`               in z
 -  `m5_g:`                some simple measure of magnitude (brightness) in a 5 pixel aperture in g band
 -  `m5_z:`                in z
 -  `fwhm:`                fitted measure of source size (failed for many sources, there 0 -- better not use this feature!)
 -  `e_fwhm:`              error/standard deviation on fwhm
 -  `colour:`              m3_g - m3_z, also known as the colour of the source.
