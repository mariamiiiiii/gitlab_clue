#! /bin/bash

rm -rf build

cd /data/cmssw/el8_amd64_gcc12/cms/cmssw/CMSSW_15_0_6
eval `scram runtime -sh`
CUDA_BASE=$(scram tool tag cuda CUDA_BASE)
TBB_BASE=$(scram tool tag tbb TBB_BASE)
BOOST_BASE=$(scram tool tag boost BOOST_BASE)
ALPAKA_BASE=$(scram tool tag alpaka ALPAKA_BASE)
cd -


cmake \
  -DCMAKE_PREFIX_PATH="${BOOST_BASE};${TBB_BASE};${CUDA_BASE}" \
  -DCHECK_CUDA_VERSION=OFF \
  -DALPAKA_DIR="${ALPAKA_BASE}" \
  -B build \
  -S . \
  -L

cmake --build build