#!/usr/bin/env bash
# ==========================================================
# CONNECT setup script
# Installs dependencies, creates conda environment,
# and configures MontePython, Cobaya, CLASS, and CAMB.
#
# Author: Andreas Nygaard (2022)
# Updated & improved (2025)
# ==========================================================

set -euo pipefail

# ------------- COLORS AND HELPERS -------------
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
CYAN="\033[0;36m"
BOLD="\033[1;35m"
NC="\033[0m"

ok()   { echo -e "${GREEN}✓${NC} $1"; }
info() { echo -e "${BLUE}→${NC} $1"; }
warn() { echo -e "${YELLOW}!${NC} $1"; }
err()  { echo -e "${RED}✗${NC} $1"; }

# ------------- FLAGS AND LOGGING -------------
AUTO_YES=false
DRYRUN=false
LOGFILE="setup_connect_$(date +%Y%m%d_%H%M%S).log"
BASEPWD="$(pwd)"
MKLROOT="${MKLROOT:-}"

for arg in "$@"; do
    case "$arg" in
        --yes|-y) AUTO_YES=true ;;
        --dry-run) DRYRUN=true ;;
        --help|-h)
            echo "Usage: $0 [--yes] [--dry-run] [--help]"
            echo
            echo "  --yes       Run non-interactively, assume defaults"
            echo "  --dry-run   Show what would be done, no changes"
            echo "  --help      Show this help and exit"
            exit 0 ;;
        *) warn "Unknown option: $arg" ;;
    esac
done

exec > >(tee -i "$LOGFILE") 2>&1

# ------------- UTILITY FUNCTIONS -------------

text_display() {
    local name="$1"
    local color="$2"
    echo -e "${color}${name}${NC}"
}

ask() {
    local __resultvar="$1"
    shift
    local prompt="$1"
    local confirm_needed="${2:-no}"
    local default="${3:-}"
    local result

    if [ "$AUTO_YES" = true ]; then
        eval "$__resultvar='$default'"
        return
    fi

    if [ "$confirm_needed" == "yes" ]; then
        while true; do
            read -p "$(text_display "${prompt} [yes/no] (default: $default): " "${BOLD}")" result
            result="${result:-$default}"  # use default if empty
            if [ "$result" == "yes" ] || [ "$result" == "no" ]; then
		break
	    fi
            echo "Please answer 'yes' or 'no'."
        done
    else
	if [ -z "$default" ]; then
            read -p "$(text_display "${prompt} (default: auto-install): " "${BOLD}")" result
	else
	    read -p "$(text_display "${prompt} (default: $default): " "${BOLD}")" result
	fi
	result="${result:-$default}"  # use default if empty
    fi
    eval "$__resultvar='$result'"
}

confirm() {
    local prompt="$1"
    local default="${2:-yes}"
    if [ "$AUTO_YES" = true ]; then return 0; fi
    while true; do
        read -p "$prompt [yes/no] (default: $default): " yn
        yn=${yn:-$default}
        case "$yn" in
            [Yy]* ) return 0 ;;
            [Nn]* ) return 1 ;;
            * ) echo "Please answer yes or no." ;;
        esac
    done
}

# ------------- INSTALL FUNCTIONS -------------

install_conda_env() {
    local env_name="$1"
    info "Creating Conda environment '${env_name}'..."
    $DRYRUN && return
    conda clean --index-cache -y || true
    conda env remove -y --name "$env_name" 2>/dev/null || true
    rm -rf "$(conda info --base)/envs/$env_name" 2>/dev/null || true
    conda create -y -n "$env_name" python=3.11
    conda activate "$env_name"
    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict
    conda install cython=3.0 scipy=1.11 numpy=1.26 astropy=5.1 pip=23.2 numexpr=2.8 pandas=2.0 \
	  matplotlib=3.7 tensorflow=2.13 tensorflow-probability=0.21.0 openmpi mpi4py==3.1.4 -y
    pip install sshkeyboard playsound
    if [[ "$OSTYPE" == "darwin"* ]]; then
        pip install -q pyobjc-core pyobjc-framework-Cocoa
    fi
    ok "Environment '$env_name' ready."
}

install_clik() {
    info "Downloading and building Planck likelihood code..."
    $DRYRUN && return
    mkdir -p resources/planck2018 && cd resources/planck2018

    for file in COM_Likelihood_Code-v3.0_R3.01.tar.gz \
                COM_Likelihood_Data-baseline_R3.00.tar.gz \
                COM_Likelihood_Data-baseline_R2.00.tar.gz; do
        curl -sL "https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=${file}" -o "$file"
        tar xf "$file"
    done

    cp -r plc_2.0 code/plc_3.0/ || true
    rm -rf plc_2.0 || true

    cd code/plc_3.0
    rm -rf plc-3.01 || true
    git clone https://github.com/benabed/clik.git plc-3.01
    cd plc-3.01
    cd "$BASEPWD"
    ok "Planck likelihood code ready."
}

install_montepython() {
    local env_name="$1"
    local install_mn="$2"
    local c_path="$3"
    local mp_path="$4"

    if [ -z "$mp_path" ]; then
	info "Cloning Monte Python into resources..."
	$DRYRUN && return
	mkdir -p resources
        info "Cloning latest Monte Python..."
        git clone https://github.com/brinckmann/montepython_public.git resources/montepython_public
        montepython_path="${PWD}/resources/montepython_public"
	mp_path="$montepython_path"
    else
        info "Using existing MontePython installation at: $mp_path"
	$DRYRUN && return
    fi

    # Patch initialise.py
    local init_file="${mp_path}/montepython/initialise.py"
    if grep -q 'and elem.find(sys.version)' "$init_file"; then
        info "Patching initialise.py to remove sys.version check..."
	sed -i.bak 's/ and elem\.find(sys\.version)\!\=\-1//' "$init_file"
        ok "Patched initialise.py successfully."
    fi

    if [ "$install_mn" = "yes" ]; then
        info "Installing MultiNest and PolyChord..."
        mkdir -p resources && cd resources
        git clone https://github.com/JohannesBuchner/MultiNest.git
        mkdir -p MultiNest/build && cd MultiNest/build
    
        CC=clang
        CXX=clang++
        FC=gfortran
        cmake -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_Fortran_COMPILER=$FC ..
    
        make -j1
        cd ../../..

	conda activate "$env_name"
        pip install pymultinest==2.12 git+https://github.com/PolyChord/PolyChordLite@master
        ok "MultiNest and PolyChord installed."
    fi

    if [ -z "$c_path" ] && [ "$setup_clik" == "yes" ]; then
        install_clik
        clik_path="${BASEPWD}/resources/planck2018/code/plc_3.0/plc-3.01/"
	c_path="$clik_path"
    fi
    if [ "$setup_clik" == "yes" ]; then
        info "Configuring Planck likelihood..."
        cd "$c_path"
	if [[ "$OSTYPE" == "darwin"* ]]; then
	    CXX=clang++ CC=clang arch -arm64 python ./waf configure --install_all_deps --cfitsio_install
	    arch -arm64 python ./waf install
	else
            ./waf configure --lapack_mkl=$MKLROOT --cfitsio_install
            ./waf install
	fi
        cd "$BASEPWD"

	setup_clik="no"
    fi

    info "Linking Monte Python..."
    cp mcmc_plugin/config_files/template.conf mcmc_plugin/config_files/connect.conf 2>/dev/null || true

    {
	echo "path['cosmo'] = '${BASEPWD}/mcmc_plugin/cosmo_wrapper'"
	echo "path['clik'] = '${c_path}'"
	echo "path['montepython'] = '${mp_path}'"
    } > mcmc_plugin/config_files/connect.conf

    cp -r mcmc_plugin/mp_likelihoods/* "$mp_path/montepython/likelihoods/" 2>/dev/null || true
    ok "Monte Python configured."
}

install_class() {
    local env_name="$1"
    local class_path="$2"

    if [ -z "$class_path" ]; then
        info "Cloning CLASS into resources..."
	$DRYRUN && return
        mkdir -p resources
        cd resources
        git clone https://github.com/lesgourg/class_public.git
        cd class_public
        git checkout 0ceb7a9a4c1e444ef5d5d56a8328a0640be91b18
        # macOS compiler patch
        if [[ "$OSTYPE" == "darwin"* ]]; then
            warn "Applying clang compiler patch for CLASS..."
            local FILE="Makefile"
            local TMP_FILE
            TMP_FILE=$(mktemp)
            while IFS= read -r line; do
                if [[ "$line" == *"CC"* && "$line" == *"="* ]]; then
                    line="${line//gcc/clang}"
                fi
                if [[ "$line" == *"CPP"* && "$line" == *"="* && "$line" != *"clang"* ]]; then
                    line="${line//g++/clang++}"
                fi
                echo "$line"
            done < "$FILE" > "$TMP_FILE"
            mv "$TMP_FILE" "$FILE"
            export CC=clang CXX=clang++ CFLAGS="-O2 -stdlib=libc++" CXXFLAGS="-O2 -stdlib=libc++" LDFLAGS="-stdlib=libc++"
            PATH=$(echo $PATH | tr ':' '\n' | grep -v gcc | grep -v g++ | paste -sd: -)
        fi
        pip install .
        cd "$BASEPWD"
    else
        info "Using existing CLASS installation: $class_path"
	$DRYRUN && return
        cd "$class_path"
        pip install .
        cd "$BASEPWD"
    fi
    ok "CLASS ready."
}

install_cobaya() {
    local env_name="$1"
    local c_path="$2"
    info "Installing Cobaya..."
    $DRYRUN && return
    conda activate "$env_name"
    pip install cobaya==3.3

    if [ -z "$c_path" ] && [ "$setup_clik" == "yes" ]; then
        install_clik
        clik_path="${BASEPWD}/resources/planck2018/code/plc_3.0/plc-3.01/"
	c_path="$clik_path"
    fi
    if [ "$setup_clik" == "yes" ]; then
	info "Configuring Planck likelihood..."
        cd "$c_path"
	if [[ "$OSTYPE" == "darwin"* ]]; then
	    CXX=clang++ CC=clang arch -arm64 python ./waf configure --install_all_deps --cfitsio_install
	    arch -arm64 python ./waf install
	else
            ./waf configure --lapack_mkl=$MKLROOT --cfitsio_install
            ./waf install
	fi
        cd "$BASEPWD"

	setup_clik="no"
    fi

    info "Configuring Cobaya..."

    if [ ! -f mcmc_plugin/config_files/connect.conf ]; then
	cp mcmc_plugin/config_files/template.conf mcmc_plugin/config_files/connect.conf
    fi
    
    mp_line=$(grep -hr "montepython" mcmc_plugin/config_files/connect.conf)
    
    {
	echo "path['cosmo'] = '${BASEPWD}/mcmc_plugin/cosmo_wrapper'"
	echo "path['clik'] = '${c_path}'"
	echo "${mp_line}"
    } > mcmc_plugin/config_files/connect.conf

    ok "Cobaya ready."
}

install_planck_lite_python() {
    info "Installing Planck lite python..."
    $DRYRUN && return
    mkdir -p resources
    cd resources
    git clone https://github.com/heatherprince/planck-lite-py.git planck_lite_py
    cd "$BASEPWD"
    ok "Planck lite python ready."
}

install_camb() {
    local env_name="$1"
    info "Installing CAMB..."
    $DRYRUN && return
    conda activate "$env_name"
    pip install --upgrade camb
    ok "CAMB ready."
}

# ------------- MAIN FLOW -------------

echo -e "\n${BLUE}--------------------------------------------------------------${NC}"
cat source/assets/logo_colour.txt 2>/dev/null || true
echo -e "${BLUE}--------------------------------------------------------------${NC}\n"
echo -e "${GREEN}CONNECT setup script${NC}\n"

create_env=""
env_name=""
setup_mp=""
montepython_path=""
MN_PC=""
cobaya=""
setup_clik=""
clik_path=""
class=""
class_path=""
camb=""

# === Conda environment ===
ask create_env "Create conda environment?" "yes" "yes"
if [ "$create_env" == "yes" ]; then
    ask env_name "Name of environment" "" "ConnectEnvironment"
else
    warn "Skipping environment creation. CONNECT will use your current environment."
    warn "You are responsible for having compatible versions of dependencies."
fi

# === MontePython ===
ask setup_mp "Use MontePython with CONNECT?" "yes" "yes"
if [ "$setup_mp" == "yes" ]; then
    ask montepython_path "Enter path to existing MontePython (path/to/montepython_public) or leave blank to auto-install" "" ""
fi
if [ "$setup_mp" == "yes" ]; then
    ask MN_PC "Install MultiNest and PolyChord?" "yes" "no"
fi

# === Cobaya ===
ask cobaya "Install Cobaya?" "yes" "yes"

# === CLIK ===
ask setup_clik "Use Planck likelihoods with CONNECT (requires CLIK)?" "yes" "yes"
if [ "$setup_clik" == "yes" ]; then
    ask clik_path "Enter path to existing clik data (path/to/code/plc_3.0/plc-3.01/) or leave blank to auto-install" "" ""
fi

# === Planck lite python ===
ask setup_planck_lite "Install Planck-lite Python package?" "yes" "yes"

# === CLASS ===
ask class "Use CLASS with CONNECT?" "yes" "yes"
if [ "$class" == "yes" ]; then
    ask class_path "Enter path to existing CLASS (/path/to/class_public) or leave blank to auto-install" "" ""
fi

# === CAMB ===
ask camb "Install CAMB?" "yes" "yes"

echo -e "\n${BLUE}--------------------------------------------------------------${NC}"
echo -e "${YELLOW}Summary of selected setup:${NC}\n"

# Default path display helpers
default_mp_path="${PWD}/resources/montepython_public"
default_class_path="${PWD}/resources/class_public"
default_clik_path="${PWD}/resources/planck2018/code/plc_3.0/plc-3.01/"

# Helper for yes/no coloring
color_flag() {
    if [ "$1" == "yes" ]; then
        echo -e "${GREEN}$1${NC}"
    else
        echo -e "${RED}$1${NC}"
    fi
}

# Helper for optional path display
path_display() {
    local answer="$1"
    local path_var="$2"
    local default_path="$3"
    if [ "$answer" == "yes" ]; then
        echo -e "(path: ${BLUE}${path_var:-$default_path}${NC})"
    else
        echo ""
    fi
}


env_display="$(text_display "$env_name" "${CYAN}")"
if [ "$create_env" == "yes" ]; then
    env_display="(${env_display})"
fi
mp_display=$(path_display "$setup_mp" "$montepython_path" "$default_mp_path")
class_display=$(path_display "$class" "$class_path" "$default_class_path")
clik_display=$(path_display "$setup_clik" "$clik_path" "$default_clik_path")

# Print summary
printf "  %-20s : %s %s\n" "Conda environment" "$(color_flag $create_env)" "$env_display"
printf "  %-20s : %s %s\n" "MontePython" "$(color_flag $setup_mp)" "$mp_display"
if [ "$setup_mp" == "yes" ]; then
    printf "  %-20s : %s\n" "MultiNest/PolyChord" "$(color_flag $MN_PC)"
fi
printf "  %-20s : %s\n" "Cobaya" "$(color_flag $cobaya)"
printf "  %-20s : %s %s\n" "CLASS" "$(color_flag $class)" "$class_display"
printf "  %-20s : %s\n" "CAMB" "$(color_flag $camb)"
printf "  %-20s : %s %s\n" "CLIK" "$(color_flag $setup_clik)" "$clik_display"
printf "  %-20s : %s\n" "Planck lite Python" "$(color_flag $setup_planck_lite)"

echo -e "\n${BLUE}--------------------------------------------------------------${NC}"

if ! confirm "Proceed with installation?"; then
    warn "Setup aborted."
    exit 0
fi

# === EXECUTION ===

# Source conda if available
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    fi
fi

# Source bashrc/profile to get conda if not already available
if [ -f ~/.bashrc ]; then
    grep -q "read -p" ~/.bashrc || source ~/.bashrc >/dev/null 2>&1
fi
if [ -f ~/.bash_profile ]; then
    grep -q "read -p" ~/.bash_profile || source ~/.bash_profile >/dev/null 2>&1
fi

# Load modules only if 'module' command exists
if command -v module >/dev/null 2>&1; then
    module load gcc openmpi cmake 2>/dev/null || true
fi


[ "$create_env" == "yes" ] && install_conda_env "$env_name"
[ "$setup_mp" == "yes" ] && install_montepython "$env_name" "$MN_PC" "$clik_path" "$montepython_path"
[ "$setup_planck_lite" == "yes" ] && install_planck_lite_python
[ "$cobaya" == "yes" ] && install_cobaya "$env_name" "$clik_path"
[ "$class" == "yes" ] && install_class "$env_name" "$class_path"
[ "$camb" == "yes" ] && install_camb "$env_name"

exec > /dev/tty

python -c "from source.assets.animate import play; play()" 2>/dev/null || true

ok "Setup complete! Log saved to ${LOGFILE}"
