	?\m??????\m?????!?\m?????	vs|??@vs|??@!vs|??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?\m???????v????Ah"lxz???Y?&S???*	gffff&@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??ݓ????!dCd?ǦQ@)a??+e??1?~eRQ@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(??y??!s??f2@)??D????1ʹ8??i0@:Preprocessing2F
Iterator::Model????镢?!??"@)?{??Pk??1fZuʹ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???????!??N?-W@)?
F%u??1P1???b@:Preprocessing2U
Iterator::Model::ParallelMapV2/?$???!NmjS?? @)/?$???1NmjS?? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??ׁsF??!??????)??ׁsF??1??????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceF%u?{?!
?R?0??)F%u?{?1
?R?0??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapj?q?????!?E(B?Q@)?????w?1OO1?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 14.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9vs|??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??v??????v????!??v????      ??!       "      ??!       *      ??!       2	h"lxz???h"lxz???!h"lxz???:      ??!       B      ??!       J	?&S????&S???!?&S???R      ??!       Z	?&S????&S???!?&S???JCPU_ONLYYvs|??@b 