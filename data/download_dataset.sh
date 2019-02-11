wget 'https://www.dropbox.com/s/nljxhtct5d7285h/Brueghel.zip?dl=0'
unzip Brueghel.zip?dl=0
rm Brueghel.zip?dl=0

wget 'https://www.dropbox.com/s/tvrt7v7m5kf1ssw/Ltll.zip?dl=0'
unzip Ltll.zip?dl=0
rm Ltll.zip?dl=0

wget 'http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz'
mkdir Oxford5K
mkdir Oxford5K/img
tar xvf oxbuild_images.tgz -C Oxford5K/img
rm oxbuild_images.tgz

