echo "Updating bert.cpp submodule..."
git submodule update --init --recursive

echo "Updating ggml submodule..."
cd bert.cpp
git submodule update --init --recursive

echo "Downloading MiniLM-L6-v2 model..."
pip3 install -r requirements.txt
python3 models/download-ggml.py download all-MiniLM-L6-v2 q4_0

echo "Building bert.cpp..."
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
make

echo "Building glue.cpp..."
cd ../..
make


