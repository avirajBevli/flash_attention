#!/bin/bash

echo "WHICH PYTHON: "
which python

python build_lib.py
echo -e "  ============ [ built lib ] ============"

python bench.py
echo -e "  ============ [ ran bench ] ============"