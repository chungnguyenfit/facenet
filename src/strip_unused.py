from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
import tensorflow as tf
import os

# set the appropriate input and output nodes
input_node_names = ['input', 'phase_train']
output_node_names = ['embeddings']

# set the appropriate path to the frozen graph and directory to output stripped graph
frozen_graph_path = '/Users/thien/git/facenet/models/facenet/20190124-163607/20190124-163607.pb'
frozen_graph_stripped_path = '/Users/thien/git/facenet/models/facenet/20190124-163607/optimized_20190124-163607.pb'


# load the frozen file and parse it to get the unserialized graph_def
restored_graph_def = None
with tf.gfile.GFile(frozen_graph_path, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())

gdef = strip_unused_lib.strip_unused(
        input_graph_def = restored_graph_def,
        input_node_names = input_node_names,
        output_node_names = output_node_names,
        placeholder_type_enum = [dtypes.float32.as_datatype_enum, dtypes.bool.as_datatype_enum] )

with tf.gfile.GFile(frozen_graph_stripped_path, "wb") as f:
    f.write(gdef.SerializeToString())
    print("Stripped frozen graph file: {}".format(frozen_graph_stripped_path))
print(" File size: {} MiB".format(os.path.getsize(frozen_graph_stripped_path) >> 20))