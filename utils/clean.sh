# sed 's# 0-1$# 0-1\n#g; s# 0-1\n$# 0-1\n\n#g; s# 1-0$# 1-0\n#g; s# 1-0\n$# 1-0\n\n#g; s# 1/2-1/2$# 1/2-1/2\n#g; s# 1/2-1/2\n$# 1/2-1/2\n\n#g' processed/output_2015-01.pgn > output_2015-01.pgn

sed 's# 0-1\[# 0-1\n\n\[#g; s# 1-0\[# 1-0\n\n\[#g; s# 1/2-1/2\[# 1/2-1/2\n\n\[#g;' processed/output_2015-01.pgn > cleaned/output_2015-01.pgn
