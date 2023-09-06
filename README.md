# hebbian-skin-cancer-segmentation

We explore bio-inspired training solutions based on the Hebbian principle
for deep learning applications in the context of image segmentation tasks.

## Setup Environment	

This tutorial shows how to setup a python environment with the exact library versions. 
The tools shown here are `asdf` and `virtualenv`

### Install asdf

Install asdf using git

	git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch v0.12.0`

Then add to your .bashrc or .zshrc file

	. "$HOME/.asdf/asdf.sh"
	. "$HOME/.asdf/completions/asdf.bash"

For the changes to have an effect restart the shell.

### Install Python

Then install `python 3.11.4`

	asdf plugin add python
	asdf install python 3.11.4

Add to `.tool-versions`

	echo "python 3.11.4" >> .tool-versions

### Create virtualenv

Upgrade `pip` and install `virtualenv`

	pip install pip --upgrade
	pip install virtualenv

Create new virtual environment

	virtualenv venv

Activate the environment

	source venv/bin/activate

Install libraries

	pip install -r requirements.txt

## Usage

1. `cd data`
2. `./prepare_data.sh`
3. `cd ..`
4. `./reproduce.sh`

## Datasets
[Brain MRI images](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)  
[Skin Lesion Segmentation](https://www.fc.up.pt/addi/ph2%20database.html)

## Requirements
Use `requirements.txt`

## Contacts
Gabriele Lagani: gabriele.lagani@phd.unipi.it

