	 9a?hA`@ 9a?hA`@! 9a?hA`@	^??? ???^??? ???!^??? ???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ 9a?hA`@i??+I???Ak?) :`@Y?x?&1??*	"??~jS@2S
Iterator::Model::ParallelMap\??b?ŗ?!??1?pk>@)\??b?ŗ?1??1?pk>@:Preprocessing2F
Iterator::Model?X?Х?!-???l?K@)?Y?Nܓ?1?X?ii9@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat?Ljh???!'?N??6@)?C???X??1?T????4@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate/???0??!}%??U?.@)??	????1????0~$@:Preprocessing2?
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice3?ۃp?!=I?J?@)3?ۃp?1=I?J?@:Preprocessing2X
!Iterator::Model::ParallelMap::Zipg)YNB??!?l;?F@)????Wn?1e????i@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor??0XrU?!u(?HUq??)??0XrU?1u(?HUq??:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap???<??!???n??0@)*??g\8P?1N?R?F???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	i??+I???i??+I???!i??+I???      ??!       "      ??!       *      ??!       2	k?) :`@k?) :`@!k?) :`@:      ??!       B      ??!       J	?x?&1???x?&1??!?x?&1??R      ??!       Z	?x?&1???x?&1??!?x?&1??JCPU_ONLY