HelloWorld
==========

create_batches_meta.py  
Generate the batches.meta
Usage: python create_batches_meta.py train.csv BATCH_DIR

create_data_batch.py 
Generate the data_batch_x (x = 1,2...,). Each batch has 4096 images.
Usage: python create_data_batch.py train.csv BATCH_DIR

create_test_batch.py
Generate the test batch, which we name the data_batch_1000.
Usage: python create_test_batch.py fer2013/fer2013.csv BATCH_DIR
