DATE="2017-12"

zstd -d lichess_db_standard_rated_$DATE.pgn.zst
rm lichess_db_standard_rated_$DATE.pgn.zst
python3 data_clean.py $DATE
# rm lichess_db_standard_rated_$DATE.pgn
