# Kaizen
To build zkPoT, first decompress 3rd.zip. Then make ./build.sh executable by using:
> chmod +x build.sh

Then run it by calling :

> ./build.sh

For zkPoT run one of the scripts given by running:
> ./{Architecture}_test.sh

Each script calls ./main {MODEL} {Number of batches} {Number of input filters} {Number of levels of SHA hashes} {PC type}

For {Number of levels of SHA hashes}, -1 corresponds to no SHA hashes. For {PC type}, 1 corresponds to Orion and 2 to Virgo.
