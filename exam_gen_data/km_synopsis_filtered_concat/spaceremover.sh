#!/bin/bash
for file in *.txt; do
	newfilename="${file// /}"
	mv "$file" "$newfilename"
done
