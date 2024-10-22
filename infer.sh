### Path to your CANDLEized model's main Python script ###
# CANDLE_MODEL=/usr/local/GraphDRP/graphdrp_baseline_pytorch.py
CANDLE_MODEL=HiDRA_infer_improve.py
# CANDLE_MODEL=frm_infer_candle.py  # TODO: change this var name!

# Path to directory containing model executable
# ??
IMPROVE_MODEL_DIR=${IMPROVE_MODEL_DIR:-$( dirname -- "$0" )}

# Check if executable exists
CANDLE_MODEL=${IMPROVE_MODEL_DIR}/${CANDLE_MODEL}
if [ ! -f ${CANDLE_MODEL} ] ; then
	echo No such file ${CANDLE_MODEL}
	exit 404
fi

if [ $# -lt 2 ] ; then
        echo "Illegal number of parameters"
        echo "CUDA_VISIBLE_DEVICES and CANDLE_DATA_DIR are required"
        exit -1
fi

if [ $# -eq 2 ] ; then
        CUDA_VISIBLE_DEVICES=$1 ; shift
        CANDLE_DATA_DIR=$1 ; shift
        CMD="python ${CANDLE_MODEL}"
        echo "CMD = $CMD"

elif [ $# -ge 3 ] ; then
        CUDA_VISIBLE_DEVICES=$1 ; shift
        CANDLE_DATA_DIR=$1 ; shift

        # if original $3 is a file, set candle_config and passthrough $@
        ### if [ -f $CANDLE_DATA_DIR/$1 ] ; then
        if [ -f $1 ] ; then
		echo "$CANDLE_DATA_DIR/$1 is a file"
                CANDLE_CONFIG=$1 ; shift
                CMD="python ${CANDLE_MODEL} --config_file $CANDLE_CONFIG $@"
                echo "CMD = $CMD $@"

        # else passthrough $@
        else
		echo "$1 is not a file"
                CMD="python ${CANDLE_MODEL} $@"
                echo "CMD = $CMD"

        fi
fi


# Display runtime arguments
echo "using CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
echo "using CANDLE_DATA_DIR ${CANDLE_DATA_DIR}"
echo "using CANDLE_CONFIG ${CANDLE_CONFIG}"

# Set up environmental variables and execute model
echo "running command ${CMD}"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} IMPROVE_DATA_DIR=${CANDLE_DATA_DIR} CANDLE_DATA_DIR=${CANDLE_DATA_DIR} $CMD
