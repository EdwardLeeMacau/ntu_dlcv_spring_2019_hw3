# TODO: create shell script for running your improve UDA model

# Download the model
if ! [ -f "./ADDA_mnistm.pth" ]; then
    wget -O ./ADDA_mnistm.pth https://www.dropbox.com/s/hxe8iwzl4hproqx/DANN_mnistm.pth?dl=0
fi
if ! [ -f "./ADDA_svhn.pth" ]; then
    wget -O ./ADDA_svhn.pth https://www.dropbox.com/s/mxhsn9jvknp4u0n/DANN_svhn.pth?dl=0
fi
if ! [ -f "./ADDA_usps.pth" ]; then
    wget -O ./ADDA_usps.pth https://www.dropbox.com/s/ru8j3thet47igrd/DANN_usps.pth?dl=0
fi

python3 predict.py --dataset $1 $2 --output $3
