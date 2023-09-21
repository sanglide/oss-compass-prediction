: '
这个脚本用于删除oss-compass-result文件夹的数据(也就是经过预处理的数据)
'

cd ../oss-compass-result || exit

rm -f repo_list.csv
rm -f label.csv

cd raw || exit
rm -rf *
cd ../segment_data || exit
rm -rf *
cd ../segment2 || exit
rm -rf *

cd ../features || exit
rm -rf *

echo "oss-compass-result文件夹已清空"