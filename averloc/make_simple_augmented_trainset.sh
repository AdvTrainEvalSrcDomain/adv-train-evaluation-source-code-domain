for i in `seq 8`
do
echo "Processing tf${i}"
sed -n -f ${1}/sed_cmd${i}.txt ${2}/data${i}.test.c2s > ${2}/adv/data${i}.test.c2s
done