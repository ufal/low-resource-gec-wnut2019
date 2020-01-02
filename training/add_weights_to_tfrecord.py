import difflib
import os
import tensorflow as tf
from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder


def get_weights(err, cor, not_equal_weight = 3):
  weights = []
  matcher = difflib.SequenceMatcher(None, cor.split(), err.split())
  for tag, i1, i2, j1, j2 in matcher.get_opcodes():
    if tag == "equal":
      for x in range(i2, i1, -1):
        weights.append(1)
    elif tag != "insert":
      for x in range(i2, i1, -1):
        weights.append(not_equal_weight)
  return weights


def _int64_feature(values):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def serialize_example(original, corrected, weights):
  """
  Creates a tf.Example message ready to be written to a file.
  """

  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.

  feature = {
    'inputs': _int64_feature(original),
    'targets': _int64_feature(corrected),
    'weights': _int64_feature(weights)
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def main(args):
  subword_encoder = SubwordTextEncoder(args.vocab_file)

  record_iterator = tf.python_io.tf_record_iterator(path=args.path)
  record_basename = os.path.basename(args.path)

  with tf.python_io.TFRecordWriter(os.path.join(args.outdir, record_basename)) as writer:
    for string_record in record_iterator:
      example = tf.train.Example()
      example.ParseFromString(string_record)

      inputs = dict(example.features.feature)['inputs']

      inputs = inputs.int64_list.value
      inputs_as_int64list = inputs
      inputs = subword_encoder.decode(inputs)

      targets = dict(example.features.feature)['targets']
      targets = targets.int64_list.value
      targets_as_int64_list = targets
      targets = subword_encoder.decode(targets)

      weights = get_weights(inputs, targets, args.weight)
      example_proto = serialize_example(inputs_as_int64list, targets_as_int64_list, weights)
      writer.write(example_proto)


if __name__ == "__main__":
  import argparse

  # Define and parse program input
  parser = argparse.ArgumentParser()
  parser.add_argument("path", help="Path to TFRecord to add weights to..")
  parser.add_argument("outdir", help="Path to directory where to store modified TFRecord file.")
  parser.add_argument("vocab_file", help="Path to vocabulary file to use.")
  parser.add_argument("weight", type=int, help="Edit weight.")
  args = parser.parse_args()
  main(args)
