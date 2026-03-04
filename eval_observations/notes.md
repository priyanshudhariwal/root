# Steps followed for exercises 1 and 2

## Configuring and Installing:
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

## Familiarisation process
I first ran the TMVA_CNN_Classification.C tutorial via
```
root -l tutorials/machine_learning/TMVA_CNN_Classification.C
```
This ran the tutorial on the CPU successfully and opened the web canvas for me with the output.

## AI Usage
As mentioned in the CERN HSF instructions that AI usage is allowed, I wish to be honest with what level of AI assistance was used in completing these tasks.

All the text written in this MD file was completely written by me without any AI generation.

The build process was done completely by me by following the instructions from CMAKE tutorials and the ROOT build from source tutorial.

During the exploration of the codebase i faced vscode showing errors in places, I used AI to help me understand those errors and install and configure the clangd language server so that there are no red herrings and i do not waste time on errors which are not there.