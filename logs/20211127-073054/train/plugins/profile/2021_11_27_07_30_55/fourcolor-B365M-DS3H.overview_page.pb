?	K:???*`@K:???*`@!K:???*`@	?Z??%???Z??%??!?Z??%??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$K:???*`@B???8a??A??A{u$`@YE???????*	?z?G?V@2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat?)V?ܞ?!??G??@@)?B;?Y???1?Q????@:Preprocessing2F
Iterator::Modelu?????!?BO?cpC@)g??j+???1t2Z??9@:Preprocessing2?
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?{b?*??!?????0@)?{b?*??1?????0@:Preprocessing2S
Iterator::Model::ParallelMap?@??!???G *@)?@??1???G *@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate???H????!?T??_7@)??]?px?1v:??T@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip_?"??]??!??\??N@)zލ?Ai?1r׷?5@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor?p?a??S?!??yvN??)?p?a??S?1??yvN??:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMapO=?ඖ?!7?f?x8@)?!9??UP?1?(^?+???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	B???8a??B???8a??!B???8a??      ??!       "      ??!       *      ??!       2	??A{u$`@??A{u$`@!??A{u$`@:      ??!       B      ??!       J	E???????E???????!E???????R      ??!       Z	E???????E???????!E???????JCPU_ONLY2black"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: 