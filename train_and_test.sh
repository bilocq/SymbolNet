#!/bin/bash

# Trains an instance of SymbolNet on uppercase letters, then performs pairing and classification tests separately on uppercase letters
# and digits. Memory of all uppercase letter and digit classes is registered before and used for the classification tests, using K-means
# with 10 centroids computed over 500 examples for each class.
#
# If the EMNIST dataset is not already present in ./data, then it will get automatically downloaded there.
#
# Compatible with mac/linux, not windows.


### VIRTUAL ENVIRONMENT
# This assumes you have created a virtual environment in ./venv and installed the requirements (pip install -r requirements.txt).
# You can remove this line if your system installation of Python already has all the requirements.
source ./venv/bin/activate


### SETTINGS
data_path="$SLURM_TMPDIR/emnist"
training_samples=500000
pairing_samples=30000
classifying_samples=3000
training_classes='uppercases'


### EXPERIMENT FOLDER
folder='./Train_and_Test_Output'
i=2
# Make experiment folder with a new name
while [ -d $folder ]
do
	folder="./Train_and_Test_Output_$i"
	i=$(( $i + 1 ))
done
mkdir $folder


### TRAINING 
python train.py --folder $folder --data_path $data_path --num_samples $training_samples --classes $training_classes --verbose_training

### PAIRING TESTS
# First we test on uppercase letters.
# Then we test on digits (which by default aren't used in training).
python pairing_test.py --folder $folder --data_path $data_path --num_samples $pairing_samples --classes uppercases
python pairing_test.py --folder $folder --data_path $data_path --num_samples $pairing_samples --classes digits

### REGISTER MEMORY
python register_memory.py --folder $folder --data_path $data_path --classes digits_uppercases --num_mem 10 --num_see 500

### CLASSIFICATION TESTS
# First we test on uppercase letters.
# Then we test on digits (which by default aren't used in training).
python classification_test.py --folder $folder --data_path $data_path --num_images $classifying_samples --classes uppercases --use_memory
python classification_test.py --folder $folder --data_path $data_path --num_images $classifying_samples --classes digits --use_memory
