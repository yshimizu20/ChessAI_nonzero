YEAR="2017"
MONTH="11"

# echo "processed/output_$YEAR-$MONTH.pgn cleaned/output_$YEAR-$MONTH.pgn"
sed 's# 0-1\[# 0-1\n\n\[#g; s# 0-1\n$# 0-1\n\n#g; s# 1-0\[# 1-0\n\n\[#g; s# 1-0\n$# 1-0\n\n#g; s# 1/2-1/2\[# 1/2-1/2\n\n\[#g; s# 1/2-1/2\n$# 1/2-1/2\n\n#g; s# 0-1$# 0-1\n#g; s# 0-1\n$# 0-1\n\n#g; s# 1-0$# 1-0\n#g; s# 1-0\n$# 1-0\n\n#g; s# 1/2-1/2$# 1/2-1/2\n#g; s# 1/2-1/2\n$# 1/2-1/2\n\n#g' processed/output_$YEAR-$MONTH.pgn > cleaned/output_$YEAR-$MONTH.pgn

python3 process.py $YEAR $MONTH
