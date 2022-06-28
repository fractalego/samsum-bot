g++ src/DequantizeAndLinear.h src/DequantizeAndLinear.cpp -o dequantise_and_linear.so\
    -I /usr/include/x86_64-linux-gnu/ \
    -I /usr/local/cuda-11.6/targets/x86_64-linux/include/ \
    -shared
