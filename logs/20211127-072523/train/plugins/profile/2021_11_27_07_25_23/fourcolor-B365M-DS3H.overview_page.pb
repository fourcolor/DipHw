?	?????Kb@?????Kb@!?????Kb@	?xw@q????xw@q???!?xw@q???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?????Kb@?q6??A?????Fb@Y{?f?lt??*	??????L@2F
Iterator::Model?#?????!О??G@)??)??z??1?_?q$?@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat?+?PO??!??=ث?<@)?[t??z??1??}?K?:@:Preprocessing2S
Iterator::Model::ParallelMap?}?֤ۂ?!3_gt?/@)?}?֤ۂ?13_gt?/@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate3?f?Ӄ?!]???<?0@)?P?l|?1??]?{?'@:Preprocessing2X
!Iterator::Model::ParallelMap::ZipXV???n??!0ax??{J@)Pqx??i?1<??	??@:Preprocessing2?
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?&?|?g?!?uFw?o@)?&?|?g?1?uFw?o@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor??_vOV?!???m?@)??_vOV?1???m?@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMapT9?)9'??!"?D???2@)2t??R?1Q?E?d??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?q6???q6??!?q6??      ??!       "      ??!       *      ??!       2	?????Fb@?????Fb@!?????Fb@:      ??!       B      ??!       J	{?f?lt??{?f?lt??!{?f?lt??R      ??!       Z	{?f?lt??{?f?lt??!{?f?lt??JCPU_ONLY2black"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: 