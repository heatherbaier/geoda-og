FOLDERS=`ls ../imagery/`

echo $FOLDERS

# for i in $FOLDERS
for i in EG GH HN HT LB NG NM PH RW TJ
do 
     echo $i
     python3 calc_country_lcs.py --iso $i
done

python3 merge_lcs.py



