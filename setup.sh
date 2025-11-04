conda create -n minerl python=3.7 pip py-opencv cudatoolkit=10.2 pytorch=1.9 -c pytorch -c conda-forge -y
conda activate minerl
conda install -c conda-forge openjdk=8 -y
export JAVA_HOME=$(dirname $(dirname "$(readlink -f "$(which java)")"))
echo "JAVA_HOME=$JAVA_HOME"
pip install git+https://github.com/minerllabs/minerl
pip install coloredlogs matplotlib aicrowd-api aicrowd-gym gym3 attrs