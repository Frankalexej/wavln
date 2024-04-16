runs=("1 2 3" "a b c d")
ms=('ae' 'aefixemb')

for m in "${!ms[@]}"; do
    for r in ${runs[m]}; do
        echo "$r" "${ms[m]}"
    done
done

# AR=('foo' 'bar' 'baz' 'bat')
# for i in "${!AR[@]}"; do
#   printf '${AR[%s]}=%s\n' "$i" "${AR[i]}"
# done

    # for r in "${runs[m]}"; do
    #     echo "$r"
    # done