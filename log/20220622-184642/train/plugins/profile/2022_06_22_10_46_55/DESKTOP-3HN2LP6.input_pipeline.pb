	??<,Ԛ????<,Ԛ??!??<,Ԛ??	޻?? ?@޻?? ?@!޻?? ?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??<,Ԛ??5?8EGr??AB?f??j??YC??6??*	gfffff^@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatM??St$??!_Cy??B@)???3???1?????@@:Preprocessing2F
Iterator::Model??ZӼ???!l(????@@)????o??1??k(?9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate=?U?????!?k(??3@)?N@aÓ?1?5??P?/@:Preprocessing2U
Iterator::Model::ParallelMapV2M?O???!?k(??? @)M?O???1?k(??? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipM?O???!?k(???P@)?&S???1??k(?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorn??t?!?5??P@)n??t?1?5??P@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicea2U0*?s?!Cy?5?@)a2U0*?s?1Cy?5?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?A`??"??!??????5@)a2U0*?c?1Cy?5???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 9.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9޻?? ?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	5?8EGr??5?8EGr??!5?8EGr??      ??!       "      ??!       *      ??!       2	B?f??j??B?f??j??!B?f??j??:      ??!       B      ??!       J	C??6??C??6??!C??6??R      ??!       Z	C??6??C??6??!C??6??JCPU_ONLYY޻?? ?@b 