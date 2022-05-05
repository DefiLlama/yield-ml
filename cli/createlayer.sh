#!/bin/bash

if [ "$1" != "" ] || [$# -gt 1]; then
	echo "Creating layer compatible with python version $1"
    # install packages
	docker run -v "$PWD":/var/task "lambci/lambda:build-python$1" /bin/sh -c "pip install -r requirements.txt -t python/lib/python$1/site-packages/; exit"
    # prune packages (remove stuff which isn't required for inference)
    docker run -v "$PWD":/var/task "lambci/lambda:build-python$1" /bin/sh -c "find . -name '*.so' | xargs strip && find . -name '*.dat' -delete && find . -name '*.npz' -delete; exit"

	zip -r layer.zip python > /dev/null
	rm -r python
	echo "Done creating layer!"
	ls -lah layer.zip

else
	echo "Enter python version as argument - ./createlayer.sh 3.6"
fi