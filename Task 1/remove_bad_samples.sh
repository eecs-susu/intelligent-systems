declare -a arr=(
    "images/rotated-squares/3.bmp"
    "images/rotated-squares/2.bmp"
    "images/rotated-squares/6.bmp"
    "images/rotated-squares/9.bmp"
    "images/rotated-squares/8.bmp"
    "images/broken-lines/5.bmp"
    "images/broken-lines/8.bmp"
    "images/rotated-rectangles/1.bmp"
    "images/rotated-rectangles/6.bmp"
    "images/right-triangles/2.bmp"
    "images/right-triangles/9.bmp"
    "images/right-triangles/7.bmp"
    )

for i in "${arr[@]}"
do
   rm $i
done