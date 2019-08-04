#!/usr/bin/env sh
set -e
source_file='web/src/lib.js'
echo "md5sums should match for source file '${source_file}' and toggle(toggle('${source_file}'))"
md5sum $source_file
./scripts/toggle_asserts.py $source_file /dev/stdout | ./scripts/toggle_asserts.py /dev/stdin /dev/stdout | md5sum -
