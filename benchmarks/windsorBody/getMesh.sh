#!/bin/sh
# Download and unpack grid
wget https://upstreamcfdcom-my.sharepoint.com/:u:/g/personal/louis_fliessbach_upstream-cfd_com/EY_GPJhIOrZNk0qZMbUlS14B46rY1aIIFrccK9qsF41LGg?download=1 -O polyMesh.tar.gz
tar -xvzf polyMesh.tar.gz
mv polyMesh constant
rm -rf polyMesh.tar.gz