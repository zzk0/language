# change the 1 to the card id that you want to detect
# change Tesla to the card name
cur0=$(nvidia-smi | grep -A 1 '1  Tesla' | tail -n 1 | awk '{print $9}' | sed 's/...$//')
cur=$(nvidia-smi | grep -A 1 '1  Tesla' | tail -n 1 | awk '{print $10}' | sed 's/...$//')

echo $cur0 $cur

if [ '$cur' -gt 0 ] 2>/dev/null ;
then
    echo 'current usage:' $cur
else
    echo 'assign cur equal to cur0:' $cur0
    cur=$cur0
fi

echo $cur
if [ $cur -le 200 ]
then
    echo 'less'
else
    echo 'more'
fi

