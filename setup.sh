#!/bin/bash

for ARGUMENT in "$@"
do
	KEY=$(echo $ARGUMENT | cut -f1 -d=)
	KEY_LENGTH=${#KEY}
	VALUE="${ARGUMENT:$KEY_LENGTH+1}"
	export "$KEY"="$VALUE"
done

function install_python () {
	if [ "$OS" = "mac" ]; then
		brew install python@3.9
	elif [ "$OS" = "vcluster" ]; then
		module load python/3.9.10
	elif [ "$OS" = "narval" ]; then
		module load python/3.9.6
	fi
}

function install_env () {
	install_python
	if [ "$OS" = "mac" ]; then
		python3.9 -m venv $ENV_NAME-env
		source $ENV_NAME-env/bin/activate
		pip install --upgrade pip

	elif [ "$OS" = "vcluster" ]; then
		python -m venv $ENV_NAME-env
		source $ENV_NAME-env/bin/activate
		python -m pip install --upgrade pip

	elif [ "$OS" = "narval" ]; then
		python -m venv $ENV_NAME-env
		source $ENV_NAME-env/bin/activate
		pip install --upgrade pip

	elif [ "$OS" = "cirrus" ]; then
		python -m venv $ENV_NAME-env
		source $ENV_NAME-env/bin/activate
		mkdir -p $HOME/tmp
		export TMPDIR=$HOME/tmp
		python -m pip install --upgrade pip

	fi
}

function install_ml_libraries () {
	if [ "$OS" = "mac" ]; then
		# Installs torch for the mac.
		pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --no-cache-dir

		# Installs tensorflow cpu on mac.
		# Tensorflow 2.10 cannot recognize the cublas library.
		# https://github.com/google-research/multinerf/issues/47#issuecomment-1258045656
		pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-2.9.2-cp39-cp39-macosx_10_14_x86_64.whl

		# Installs jax for cpu on mac.
		pip install --upgrade "jax[cpu]"

	elif [ "$OS" = "vcluster" ]; then
		python -m pip install tensorflow-gpu==2.11.0
		python -m pip install torch torchvision torchaudio

	elif [ "$OS" = "cirrus" ]; then
		python -m pip install tensorflow-gpu==2.11.0

		python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

	elif [ "$OS" = "narval" ]; then
		python -m pip install --no-index tensorflow==2.11.0
		python -m pip install --no-index torch torchvision torchtext torchaudio
		module load StdEnv/2020  gcc/9.3.0  cuda/11.4 arrow/11.0.0
		python -m pip install pyarrow
		python -m pip install datasets
	fi
}

function install_prompt_package () {
	if [ "$DEV" = "true" ]; then
		# Installs pre-commit tools as well.
		python -m pip install -e .'[dev]'

	elif [ "$DEV" = "false" ]; then
		pip install .

	fi

}

function install_reference_methods () {
	if [ "$ENV_NAME" = "prompt_torch" ]; then
		python -m pip install transformers datasets sentencepiece nltk
		python -m pip install evaluate bert-score supar pandas scikit-learn tensorboard absl-py peft

	fi

}

install_env
install_ml_libraries
install_prompt_package
install_reference_methods
