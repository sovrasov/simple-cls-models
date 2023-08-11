git clone https://github.com/teavanist/MNIST-JPG.git
mkdir -p data/MNIST
cd data/MNIST
ls -l ../../MNIST-JPG
unzip ../../MNIST-JPG/'MNIST Dataset JPG format.zip'
mv 'MNIST Dataset JPG format'/'MNIST - JPG - testing' ./val
mv 'MNIST Dataset JPG format'/'MNIST - JPG - training' ./train
rm -r 'MNIST Dataset JPG format'
cd -
rm -rf MNIST-JPG