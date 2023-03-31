# test/inference script

# create inference data list and convert to datasets format
python extract_data_test.py > extract_data_test.log
python create_datasets_test.py > create_datasets_test.log
# test the model performance with best checkpoint (default batch size 32)
python test_fast.py > test_fast.log
echo "Test/inference script finished"