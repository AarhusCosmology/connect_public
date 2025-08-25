#!/bin/bash

# This script installs all dependencies and sets up a conda environment.
# The user must have the loadable kernel modules 'intel' and 'mkl' installed
#
# Author: Andreas Nygaard (2022)



NO='\033[0;31mno\033[0m'
YES='\033[0;32myes\033[0m'

echo -e "--------------------------------------------------------------\n\n"
cat source/assets/logo_colour.txt
echo -e "\n--------------------------------------------------------------\n"

echo -e "Running setup script for connect\n"

echo -e "The following things can be done (but can also be skipped):"
echo -e "    - create conda environment with all dependencies"
echo -e "    - install and setup Monte Python or use previous installation"
echo -e "    - install and setup CLASS or use previous installation"
echo -e "    - install Cobaya and CAMB (in environment)\n\n"
echo -e "--------------------------------------------------------------\n\n"


while [ -z $create_env ]
do
    echo "Creating conda environment. Proceed? [yes, skip]"
    read create_env
done

if [ $create_env == "yes" ]
then
    echo "Enter name of conda environment to create, or leave blank to use"
    echo "default name 'ConnectEnvironment':"
    read env_name
    if [ -z $env_name ]
    then
	env_name="ConnectEnvironment"
    fi
fi

if ! [ $create_env == "yes" ]
then
    echo "Enter name of conda environment to use, or leave blank to not use"
    echo "an environment:"
    read env_name
fi

while [ -z $setup_mp ]
do
    echo "Do you want to use Monte Python with CONNECT? [yes, no]"
    read setup_mp
    echo "Do you want to install MultiNest and PolyChord? [yes, no]"
    read MN_PC
done

if [ $setup_mp == "yes" ]
then
    echo "Enter absolute path to montepython_public, or leave blank to"
    echo "download and install it here:"
    read montepython_path
    echo "Enter absolute path to clik data (../code/plc_3.0/plc-3.01/),"
    echo "or leave blank to download and install it here:"
    read clik_path
fi

while [ -z $cobaya ]
do
    echo "Do you want to install Cobaya in the environment? [yes, no]"
    read cobaya
done

if [ -z $clik_path ] && [ $cobaya == "yes" ] && [ $setup_mp != "yes" ]
then
    echo "Enter absolute path to clik data (../code/plc_3.0/plc-3.01/),"
    echo "or leave blank to download and install it here:"
    read clik_path
fi

while [ -z $class ]
do
    echo "Do you want to install CLASS in the environment? [yes, no]"
    read class
done

if [ $class == "yes" ]
then
    echo "If you already have a CLASS installation, enter the absolute"
    echo "path. Otherwise, leave blank and CLASS repo will be cloned:"
    read class_path
fi

while [ -z $camb ]
do
    echo "Do you want to install CAMB in the environment? [yes, no]"
    read camb
done

echo -e "\n--------------------------------------------------------------\n"

if [ $create_env == "yes" ]
then
    Ans1=$YES
    env_name_string="\n    Name of conda environment:\n    \033[0;34m${env_name}\033[0m"
else
    Ans1=$NO
fi
if [ $setup_mp == "yes" ]
then
    Ans2=$YES
    if [ $MN_PC == "yes" ]
    then
	Ans2_1=$YES
    else
	Ans2_1=$NO
    fi
    if ! [ -z $clik_path ]
    then
	clik_mp="\n    Path to clik:\n    \033[0;34m${clik_path}\033[0m"
    else
	clik_mp="\n    Downloading clik to \033[0;34mconnect/resources\033[0m"
    fi
    if ! [ -z $montepython_path ]
    then
	if [[ $montepython_path == *montepython_public/ ]]
	then
	    montepython_path=${montepython_path::-1}
	elif [[ $montepython_path == */ ]]
	then
	    montepython_path="${montepython_path}montepython_public"
	elif ! [[ $montepython_path == *montepython_public ]]
	then
	    montepython_path="${montepython_path}/montepython_public"
	fi
	path_mp="\n    Path to Monte Python:\n    \033[0;34m${montepython_path}\033[0m"
    else
	path_mp="\n    Cloning Monte Python repo to \033[0;34mconnect/resources\033[0m"
    fi
else
    Ans2=$NO
fi

if [ $cobaya == "yes" ]
then
    Ans3=$YES
    if ! [ $Ans2 == $YES ]
    then
	if ! [ -z $clik_path ]
	then
	    clik_cobaya="\n    Path to clik:\n    \033[0;34m${clik_path}\033[0m"
	else
	    clik_cobaya="\n    Downloading clik to \033[0;34mconnect/resources\033[0m"
	fi
    fi
else
    Ans3=$NO
fi

if [ $class == "yes" ]
then
    Ans4=$YES
    if ! [ -z $class_path ]
    then
	path_class="\n    Path to CLASS:\n    \033[0;34m${class_path}\033[0m"
    else
	path_class="\n    Cloning CLASS repo to \033[0;34mconnect/resources\033[0m"
    fi
else
    Ans4=$NO
fi

if [ $camb == "yes" ]
then
    Ans5=$YES
else
    Ans5=$NO
fi

echo -e "You have selected the following:\n"
echo -e "Create conda environment             :                   ${Ans1}${env_name_string}"
echo -e "Setup link to Monte Python           :                   ${Ans2}${path_mp}${clik_mp}"
echo -e "Install MultiNest and PolyChord      :                   ${Ans2_1}"
echo -e "Install Cobaya                       :                   ${Ans3}${clik_cobaya}"
echo -e "Install CLASS                        :                   ${Ans4}${path_class}"
echo -e "Install CAMB                         :                   ${Ans5}"

echo -e "\n--------------------------------------------------------------\n"

while [ -z $proceed ]
do
    echo -e "\nProceed? [yes, abort]"
    read proceed
done

if [ $proceed == "abort" ]
then
    echo -e "\nYou have aborted the setup. Please try again\n"
    exit 0
fi



##################################################################################

##############################     Actual setup     ##############################

##################################################################################


function install_clik {
    connect_path=$PWD
    mkdir -p resources
    cd resources
    rm -rf planck2018
    mkdir planck2018
    cd planck2018
    echo "--> Downloading Planck likelihood code and data..."
    for file in COM_Likelihood_Code-v3.0_R3.01.tar.gz COM_Likelihood_Data-baseline_R3.00.tar.gz COM_Likelihood_Data-baseline_R2.00.tar.gz
    do
	echo "Handling" ${file} "..."
	curl https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=${file} --output ${file}
	tar xf ${file}
    done
    echo "--> ...done!"
    # Move 2015 data directory to the expected location
    cp -r plc_2.0 code/plc_3.0/
    rm -rf plc_2.0
    echo "--> Downloading and installing cfitsio..."
    cd code/plc_3.0/plc-3.01/
    wget http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio-3.47.tar.gz
    tar xf cfitsio-3.47.tar.gz
    
    # The gcc compiler which will be silently loaded alongside
    # the intel compiler cannot compile cfitsio, so load later.
    
    cd cfitsio-3.47
    ./configure
    make -j
    make install
    cd ..
    echo "--> ...done!"
    
    cd $connect_path
    clik_path="${PWD}/resources/planck2018/code/plc_3.0/plc-3.01/"
}


source ~/.bashrc 2> /dev/null
source ~/.bash_profile 2> /dev/null
source "$(conda info | grep -i 'base environment' | awk '{for(i=1;i<=NF;i++) if($i ~ /\//) print $i}')/etc/profile.d/conda.sh"
module load gcc openmpi cmake 2> /dev/null
conda init


if [ $Ans1 == $YES ]
then
    echo "--> Creating Conda environment, this will take a few minutes..."
    conda clean --index-cache -y
    # Remove ConnectEnvironment if it exists
    conda env remove -y --name $env_name
    conda create -y --name $env_name python=3.10 cython=3.0 scipy=1.11 numpy=1.26 astropy=5.1 pip=23.2 numexpr=2.8 pandas=2.0
    conda activate $env_name
    export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH
    pip install matplotlib==3.7
    pip install mpi4py==3.1.4
    pip install tensorflow==2.13
    pip install tensorflow-probability==0.18.0
    pip install sshkeyboard
    pip install playsound
    if [ "$Ans2_1" == "$YES" ]
    then
	pip install pymultinest==2.12
	pip install git+https://github.com/PolyChord/PolyChordLite@master
    fi
    echo "--> ..done!"
fi

if [ $Ans2 == $YES ]
then
    if [ -z $montepython_path ]
    then
	echo "--> Cloning Monte Python into resources..."
	mkdir -p resources
	cd resources
	git clone -b 3.5 https://github.com/brinckmann/montepython_public.git
	cd ..
	montepython_path="${PWD}/resources/montepython_public"
	echo "--> ..done!"
	
	if [ $Ans2_1 == $YES ]
	then
	    echo "--> Installing MultiNest..."
	    cd resources
	    git clone https://github.com/JohannesBuchner/MultiNest.git
	    rm -rf MultiNest/build/*
	    cd MultiNest/build
	    CC=gcc CXX=g++ cmake -D CMAKE_Fortran_COMPILER=gfortran ..
	    make
	    cd ../../..
	    echo "--> ...done!"
	    # Remember to add MultiNest to LD_LIBRARY_PATH when using
	    # export LD_LIBRARY_PATH=$PWD/MultiNest/lib/:$LD_LIBRARY_PATH
	fi
    fi
    
    if [ -z $clik_path ]
    then
	install_clik
    fi
    echo "--> Installing Planck likelihood code..."
    connect_path=$PWD
    cd $clik_path
    if ! [ -z $env_name ]
    then
	source activate $env_name
    fi
    ./waf configure --lapack_mkl=$MKLROOT --cfitsio_prefix=$PWD/cfitsio-3.47
    ./waf install
    cd $connect_path
    echo "-->...done!"

    echo "--> Linking Monte Python..."
    cp mcmc_plugin/connect.conf.template mcmc_plugin/connect.conf
    echo "path['cosmo'] = '${PWD}/mcmc_plugin'" > mcmc_plugin/connect.conf
    echo "path['clik'] = '${clik_path}'" >> mcmc_plugin/connect.conf
    echo "path['montepython'] = '${montepython_path}'" >> mcmc_plugin/connect.conf
    
    cp -r mcmc_plugin/mp_likelihoods/Planck_lowl_EE_connect $montepython_path/montepython/likelihoods/
    echo "--> ...done!"
fi

if [ $Ans3 == $YES ]
then
    echo "--> Installing Cobaya..."
    if ! [ -z $env_name ]
    then
        conda activate $env_name
    fi
    python -m pip install cobaya --upgrade
    echo "--> ...done!"
    if [ $Ans2 != $YES ]
    then
	if [ -z $clik_path ]
        then
	    install_clik
	fi
	echo "--> Installing Planck likelihood code..."
	connect_path=$PWD
	cd $clik_path
	./waf configure --lapack_mkl=$MKLROOT --cfitsio_prefix=$PWD/cfitsio-3.47
	./waf install
	cd $connect_path
	echo "-->...done!"
    fi
    if ! [ -f "mcmc_plugin/connect.conf" ]
    then
	cp mcmc_plugin/connect.conf.template mcmc_plugin/connect.conf
    fi
    line1="path['cosmo'] = '${PWD}/mcmc_plugin'"
    line2="path['clik'] = '${clik_path}'"
    line3=$(grep -hr "montepython" mcmc_plugin/connect.conf)
    echo $line1 > mcmc_plugin/connect.conf
    echo $line2 >> mcmc_plugin/connect.conf
    echo $line3 >> mcmc_plugin/connect.conf
fi

if [ $Ans4 == $YES ]
then
    if ! [ -z $env_name ]
    then
        conda activate $env_name
    fi
    if ! [ -z $class_path ]
    then
	echo "--> Building classy wrapper..."
	connect_path=$PWD
	cd $class_path
	make clean
	make -j
	cd $connect_path
	echo "--> ...done!"
    else
	echo "--> Cloning CLASS into resources..."
        mkdir -p resources
	cd resources
	git clone https://github.com/lesgourg/class_public.git
	cd class_public
	git checkout 0ceb7a9a4c1e444ef5d5d56a8328a0640be91b18
	echo "--> Building classy wrapper..."

	if [[ "$OSTYPE" == "darwin"* ]]; then
	    echo "Detected macOS: setting Clang for Python build"
	    FILE="Makefile"
	    # Use temporary file to avoid in-place issues across platforms
	    TMP_FILE=$(mktemp)
	    # Process the file line by line
	    while IFS= read -r line; do
		if [[ "$line" == *"CC"* && "$line" == *"="* ]]; then
		    line="${line//gcc/clang}"
		fi
		if [[ "$line" == *"CPP"* && "$line" == *"="* && "$line" != *"clang"* ]]; then
		    line="${line//g++/clang++}"
		fi
		echo "$line"
	    done < "$FILE" > "$TMP_FILE"
	    # Replace original file
	    mv "$TMP_FILE" "$FILE"
	    export CC=clang
	    export CXX=clang++
	    export CFLAGS="-O2 -stdlib=libc++"
	    export CXXFLAGS="-O2 -stdlib=libc++"
	    export LDFLAGS="-stdlib=libc++"
	    PATH=$(echo $PATH | tr ':' '\n' | grep -v gcc | grep -v g++ | paste -sd: -)
	fi
        pip install .
        cd $connect_path
        echo "--> ...done!"
    fi
fi

if [ $Ans5 == $YES ]
then
    echo "--> Installing CAMB..."
    if ! [ -z $env_name ]
    then
        conda activate $env_name
    fi
    pip install camb --upgrade
    echo "--> ...done!"
fi

python -c "from source.assets.animate import play; play()"

echo -e "\nSetup is all done!\n"
