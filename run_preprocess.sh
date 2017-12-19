for i in {1..38}
do
( python preprocess/csv_to_sequences.py --input_dirname ../data/kddcup2015/clean/train_sequences.csv/course_id\=$i/ --output_filename ../data/kddcup2015/sequences/len_ge_5/c$i.p --min_sequence_lenght 5 > ../data/kddcup2015/sequences/len_ge_5/info-c$i.log ) &
done
wait
