FILES="./space_analysis/*.bin"
for f in $FILES
do
  python3 -m memray flamegraph "$f" -o "${f}.html"
done
