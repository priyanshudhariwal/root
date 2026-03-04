# Steps followed for building and installing ROOT with SOFIE

## Configuring:
1. Create build_dir and install_dir
2. for build config:
    ```
    cmake -G Ninja -S . -B build_dir \
    -DCMAKE_INSTALL_PREFIX="$(pwd)/install_dir/" \
    -Dtmva-sofie=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    ```
3. build step:
    ```
    cmake --build build_dir -j8
    ```
4. install step:
    ```
    cmake --install build_dir
    ```
5. verification:
    ```
    source install_dir/bin/thisroot.sh
    root --version
    root-config --features | rg "sofie"
    ```
