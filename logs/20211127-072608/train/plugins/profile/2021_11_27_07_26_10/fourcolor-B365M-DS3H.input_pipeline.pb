	?el?&d{@?el?&d{@!?el?&d{@	񗳼???񗳼???!񗳼???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?el?&d{@?V|Cᳵ?A?d73?a{@Yy??.??*	5^?I?P@2F
Iterator::ModelYm?_u???!??؟g?H@)mFA????1?0I ??@@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat?y ?????!jI?m%:@)rP?Lۏ?1!?*?)7@:Preprocessing2S
Iterator::Model::ParallelMap?~m?????!??>~r/@)?~m?????1??>~r/@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate6=((E+??!s?
Hy?0@)?^(`;??1b?????(@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip??f??}??!('`?oI@)?E?n?1o?1cJ??a?@:Preprocessing2?
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?y?'Lh?!??N??@)?y?'Lh?1??N??@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor??N?j`?!vBQ?V?@)??N?j`?1vBQ?V?@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap2?CP5??!t?G+3@)D???XPX?1F5???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?V|Cᳵ??V|Cᳵ?!?V|Cᳵ?      ??!       "      ??!       *      ??!       2	?d73?a{@?d73?a{@!?d73?a{@:      ??!       B      ??!       J	y??.??y??.??!y??.??R      ??!       Z	y??.??y??.??!y??.??JCPU_ONLY