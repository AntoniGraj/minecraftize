#!/bin/bash

rm out.tif out.geojson slope.tif slope.geojson

fst_n=`echo $1 | tr -d ','`
fst_e=`echo $2 | tr -d ','`
snd_n=`echo $3 | tr -d ','`
snd_e=`echo $4 | tr -d ','`
size=$5

echo $fst_e

curl "https://mapy.geoportal.gov.pl/wss/service/PZGIK/NMT/GRID1/WCS/DigitalTerrainModelFormatTIFF?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage&FORMAT=image/tiff&COVERAGE=DTM_PL-KRON86-NH_TIFF&BBOX=$fst_e,$fst_n,$snd_e,$snd_n&CRS=EPSG:4326&RESPONSE_CRS=EPSG:4326&WIDTH=$size&HEIGHT=$size" -o out.tif
gdal_polygonize out.tif -f GeoJSON out.geojson
gdaldem slope out.tif slope.tif -s 111120 -compute_edges
gdal_polygonize slope.tif -f GeoJSON slope.geojson

#python3 get.py
