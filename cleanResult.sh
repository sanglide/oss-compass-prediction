cd oss-compass-result || exit

rm repo_list.csv
rm label.csv

cd raw || exit
# shellcheck disable=SC2035
rm -rf *
cd ..
cd segment_data || exit
# shellcheck disable=SC2035
rm -rf *
cd ..
cd segment2 || exit
# shellcheck disable=SC2035
rm -rf *
cd ..

cd ..
