git clone https://github.com/AlexeyAB/darknet
cd darknet
make

sed -i '/GPU=0/c\GPU=1' ./Makefile
sed -i '/CUDNN=0/c\CUDNN=1' ./Makefile
sed -i '/OPENCV=0/c\OPENCV=1' ./Makefile

cp cfg/yolov3.cfg cfg/yolov3-train.cfg

touch data/obj.names
touch data/obj.data

echo -e 'license-plate' > data/obj.names
echo -e 'car' > data/obj.names
echo -e 'classes = 2\ntrain = data/train.txt\nvalid = data/test.txt\nnames = data/obj.names\nbackup = /content/implementation/pretrained/weights' > data/obj.data

touch data/train
cp -r ../implementation/data/train/* ./data/train

python3 ../implementation/setup.py --data_dir ./data/ --darknet_path .

wget https://pjreddie.com/media/files/darknet53.conv.74
 