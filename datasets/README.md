# Terrain datasets

EP-high (Eaglepeak), BH-high (Bearhead), SF-high (San Francisco South), USGS-CO (Colorado), USGS-WA (Washington), and LA (author-shared 4000 x 4000 elevation array).

Each directory contains the final 50000 valid labels and 40000/5000/5000 train/validation/test query pairs. EP-high uses the historical seed-42 shuffled split; the other five retain their fixed-order split. Do not mix these protocols. Labels are approximate surface geodesic distances, not certified exact MMP ground truth.

Large input files are intended for GitHub Release `terrain-datasets-20261023`, preserving the original data after gzip decompression. File sizes and original SHA-256 hashes are in manifest.json. LA is supplied as its original elevation array; LA.off is not present in the local archive. The historical LA mesh-conversion script and metadata are included as reference, with their original directory assumptions retained.

This directory records data inputs and label/split protocols. It does not claim that all six datasets have complete baseline experiments.
