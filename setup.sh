#!/bin/bash

NO='\033[0;31mno\033[0m'
YES='\033[0;32myes\033[0m'

echo -e "--------------------------------------------------------------\n\n"
cat source/logo.txt
echo -e "\n--------------------------------------------------------------\n"

echo -e "Running setup script for connect\n"

echo -e "The following things can be done (but can also be skipped):"
echo -e "    - create conda environment with all dependencies"
echo -e "    - setup path to Monte Python and add custom likelihoods"
echo -e "    - install Cobaya, CLASS and CAMB (in environment)\n\n"
echo -e "--------------------------------------------------------------\n\n"
echo "Creating conda environment. Proceed? [yes, skip]"
read create_env
env_name="ConnectEnvironment"
if ! [ $create_env == "yes" ]
then
    echo "Enter name of conda environment to use, or leave blank to not use"
    echo "an environment:"
    read env_name
fi
echo "Do you want to use Monte Python with CONNECT? [yes, no]"
read setup_mp
if [ $setup_mp == "yes" ]
then
    echo "Enter absolute path to montepython_public, or leave blank to"
    echo "download and install it here:"
    read montepython_path
    echo "Enter absolute path to clik data (../code/plc_3.0/plc-3.01/),"
    echo "or leave blank to download and install it here:"
    read clik_path
fi

echo "Do you want to install Cobaya in the environment? [yes, no]"
read cobaya
if [ -z $clik_path ] && [ $cobaya == "yes" ] && [ $setup_mp != "yes" ]
then
    echo "Enter absolute path to clik data (../code/plc_3.0/plc-3.01/),"
    echo "or leave blank to download and install it here:"
    read clik_path
fi
echo "Do you want to install CLASS in the environment? [yes, no]"
read class
if [ $class == "yes" ]
then
    echo "If you already have a CLASS installation, enter the absolute"
    echo "path. Otherwise, leave blank and CLASS repo will be cloned:"
    read class_path
fi
echo "Do you want to install CAMB in the environment? [yes, no]"
read camb

echo -e "\n--------------------------------------------------------------\n"

if [ $create_env == "yes" ]
then
    Ans1=$YES
else
    Ans1=$NO
fi
if [ $setup_mp == "yes" ]
then
    Ans2=$YES
    if ! [ -z $clik_path ]
    then
	clik_mp="\n    Path to clik:\n    \033[0;34m${clik_path}\033[0m"
    else
	clik_mp="\n    Downloading clik to this location"
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
	path_mp="\n    Cloning Monte Python repo to this location"
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
	    clik_cobaya="\n    Downloading clik to this location"
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
	path_class="\n    Cloning CLASS repo from GitHub"
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
echo -e "Create conda environment             :                   ${Ans1}"
echo -e "Setup link to Monte Python           :                   ${Ans2}${path_mp}${clik_mp}"
echo -e "Install Cobaya                       :                   ${Ans3}${clik_cobaya}"
echo -e "Install CLASS                        :                   ${Ans4}${path_class}"
echo -e "Install CAMB                         :                   ${Ans5}"

echo -e "\n--------------------------------------------------------------\n"

echo -e "\nProceed? [yes, abort]"
read proceed
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




if [ $Ans1 == $YES ]
then
    echo "--> Creating Conda environment, this will take a few minutes..."
    conda clean --index-cache
    # Remove ConnectEnvironment if it exists
    conda env remove -y --name $env_name
    conda create -y --name $env_name
    source activate $env_name
    conda install -y cython matplotlib scipy numpy mpi4py
    conda install -c anaconda tensorflow-gpu
    python -m pip install pyfits
    echo "--> ..done!"
fi

if [ $Ans2 == $YES ]
then
    if [ -z $montepython_path ]
    then
	echo "--> Cloning Monte Python into resources..."
	mkdir -p resources
	cd resources
	git clone https://github.com/brinckmann/montepython_public.git
	cd ..
	montepython_path="${PWD}/resources/montepython_public"
	echo "--> ..done!"
    fi
    
    if [ -z $clik_path ]
    then
	install_clik
    fi
    echo "--> Installing Planck likelihood code..."
    connect_path=$PWD
    cd $clik_path
    module load mkl
    module load intel
    if ! [ -z $env_name ]
    then
	source activate $env_name
    fi
    ./waf configure --lapack_mkl=$MKLROOT --cfitsio_prefix=$PWD/cfitsio-3.47
    ./waf install
    cd $connect_path
    echo "-->...done!"

    echo "--> Linking Monte Python..."
    echo "path['cosmo'] = '${PWD}/mp_plugin'" > mp_plugin/connect.conf
    echo "path['clik'] = '${clik_path}'" >> mp_plugin/connect.conf
    echo "path['montepython'] = '${montepython_path}'" >> mp_plugin/connect.conf
    
    cp -r mp_plugin/mp_likelihoods/Planck_lowl_EE_connect $montepython_path/montepython/likelihoods/
    echo "--> ...done!"
fi

if [ $Ans3 == $YES ]
then
    echo "--> Installing Cobaya..."
    if ! [ -z $env_name ]
    then
        source activate $env_name
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
	module load mkl
	module load intel
	./waf configure --lapack_mkl=$MKLROOT --cfitsio_prefix=$PWD/cfitsio-3.47
	./waf install
	cd $connect_path
	echo "-->...done!"
    fi
    line1="path['cosmo'] = '${PWD}/mp_plugin'"
    line2="path['clik'] = '${clik_path}'"
    line3=$(grep -hr "montepython" mp_plugin/connect.conf)
    echo $line1 > mp_plugin/connect.conf
    echo $line2 >> mp_plugin/connect.conf
    echo $line3 >> mp_plugin/connect.conf
fi

if [ $Ans4 == $YES ]
then
    if ! [ -z $env_name ]
    then
        source activate $env_name
    fi
    if ! [ -z $class_path ]
    then
	echo "--> Building classy wrapper..."
	connect_path=$PWD
	cd $class_path
        module unload intel
	make clean
	make
	echo "--> ...done!"
    else
	echo "--> Cloning CLASS into resources..."
        mkdir -p resources
	cd resources
	git clone https://github.com/lesgourg/class_public.git
	cd class_public
	echo "--> Building classy wrapper..."
	make clean
	make
	cd ../..
	echo "--> ...done!"
    fi
fi

if [ $Ans5 == $YES ]
then
    echo "--> Installing CAMB..."
    if ! [ -z $env_name ]
    then
        source activate $env_name
    fi
    python -m pip install camb --upgrade
    echo "--> ...done!"
fi

echo -e "\nSetup is all done!\n"
