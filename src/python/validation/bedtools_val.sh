
inTR_output=$1
validation=$2
window=$3





awk -F'\t' -v OFS='\t' '{print $1, $2}' "$inTR_output" |
 awk -v win="$window" -F'\t' -v OFS='\t' '{$3 = $2; $3 = $3 + win; $2 = $2 - win; print}' |
 awk '!seen[$0]++' > novelty_unique.bed

overlaps=$(bedtools intersect -a novelty_unique.bed \
-b "$validation" -c |
 awk '$4 != 0 {count++} END {print count+0}')

num_unique=$(($(wc -l < novelty_unique.bed) - 1)) # -1 to account for header

echo "$overlaps / $num_unique"
