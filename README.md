# gpqhe
A C library doing fully homomorphic encryption under the license of LGPL.

# How to use

```sh
git clone git@github.com:OChicken/gpqhe.git
cd gpqhe
make
cd tests
LD_LIBRARY_PATH=$PWD/../lib:$LD_LIBRARY_PATH
make test-gpqhe
./test-gpqhe enc sk
````
