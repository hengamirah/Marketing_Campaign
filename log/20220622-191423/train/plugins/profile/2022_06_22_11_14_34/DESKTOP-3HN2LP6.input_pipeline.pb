	?ZB>?????ZB>????!?ZB>????	(?2λX@(?2λX@!(?2λX@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?ZB>?????]K?=??A??\m????YL?
F%u??*	33333Se@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatB>?٬???!Xrw???@@)??ݓ????1??M	]>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?	h"lx??!ێ??N>@)?H?}??1?w??;@:Preprocessing2F
Iterator::Model-C??6??!?{?L >@)?J?4??1ެ??C9@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipTt$?????!????Q@)?J?4??1%qk2??@:Preprocessing2U
Iterator::Model::ParallelMapV2??ǘ????!?v?0??@)??ǘ????1?v?0??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*?s?!?\?9@?@)a2U0*?s?1?\?9@?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceHP?s?r?!߸87(?@)HP?s?r?1߸87(?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap_?Qګ?!aáQ0??@)??_vOf?1SH?@?R??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9(?2λX@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?]K?=???]K?=??!?]K?=??      ??!       "      ??!       *      ??!       2	??\m??????\m????!??\m????:      ??!       B      ??!       J	L?
F%u??L?
F%u??!L?
F%u??R      ??!       Z	L?
F%u??L?
F%u??!L?
F%u??JCPU_ONLYY(?2λX@b 