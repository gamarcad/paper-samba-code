# bin/bash

VENV_NAME='samba'

# ==========================================================
# Checking necessary language and librairies that must be
# install by the user.
# ==========================================================
# Checks that python3 is installed as well as pip3
if (! type "python3" > /dev/null); then
  echo "python3 is not installed on your system."
  echo "Install it before to launch this program".
  exit 1
fi

if (! type "pip3" > /dev/null); then
  echo "pip3 is not installed on your system."
  echo "Install it before to launch this program".
  exit 1
fi


# ===========================================================
# Creating Python virtual environment
# ===========================================================
if [ ! -d $VENV_NAME ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv $VENV_NAME
  echo "Virtual environment created successfully"
else
  if [ -f $VENV_NAME ]; then
    echo "Virtual environment cannot be created: a file called $VENV_NAME already exists"
    exit 1
  elif [ ! -f $VENV_NAME/bin/python3 ]; then
    echo "Virtual environment cannot be created: Python3 binary not found inside the virtual environment"
    exit 2
  else
    echo "Virtual environment already exists"
  fi
fi

# ==========================================================
# Initializing Python virtual environment
# ==========================================================
echo "Initializing virtual environment..."

# installing requires dependencies detailed in the requirements.txt file
"$VENV_NAME/bin/pip3" install -r requirements.txt


echo '------------------------------------'
echo 'Virtual environment ready to be used.'
echo "Type this command to enter in environment:  source $VENV_NAME/bin/activate"
echo "Type this commant to exit the environment:  deactivate"



