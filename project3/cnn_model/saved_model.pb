¦
Ñ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878¦

conv_net_9/conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameconv_net_9/conv2d_24/kernel

/conv_net_9/conv2d_24/kernel/Read/ReadVariableOpReadVariableOpconv_net_9/conv2d_24/kernel*&
_output_shapes
: *
dtype0

conv_net_9/conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv_net_9/conv2d_24/bias

-conv_net_9/conv2d_24/bias/Read/ReadVariableOpReadVariableOpconv_net_9/conv2d_24/bias*
_output_shapes
: *
dtype0
¦
'conv_net_9/batch_normalization_24/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'conv_net_9/batch_normalization_24/gamma

;conv_net_9/batch_normalization_24/gamma/Read/ReadVariableOpReadVariableOp'conv_net_9/batch_normalization_24/gamma*
_output_shapes
: *
dtype0
¤
&conv_net_9/batch_normalization_24/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&conv_net_9/batch_normalization_24/beta

:conv_net_9/batch_normalization_24/beta/Read/ReadVariableOpReadVariableOp&conv_net_9/batch_normalization_24/beta*
_output_shapes
: *
dtype0
²
-conv_net_9/batch_normalization_24/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-conv_net_9/batch_normalization_24/moving_mean
«
Aconv_net_9/batch_normalization_24/moving_mean/Read/ReadVariableOpReadVariableOp-conv_net_9/batch_normalization_24/moving_mean*
_output_shapes
: *
dtype0
º
1conv_net_9/batch_normalization_24/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31conv_net_9/batch_normalization_24/moving_variance
³
Econv_net_9/batch_normalization_24/moving_variance/Read/ReadVariableOpReadVariableOp1conv_net_9/batch_normalization_24/moving_variance*
_output_shapes
: *
dtype0

conv_net_9/conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *,
shared_nameconv_net_9/conv2d_25/kernel

/conv_net_9/conv2d_25/kernel/Read/ReadVariableOpReadVariableOpconv_net_9/conv2d_25/kernel*&
_output_shapes
:  *
dtype0

conv_net_9/conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv_net_9/conv2d_25/bias

-conv_net_9/conv2d_25/bias/Read/ReadVariableOpReadVariableOpconv_net_9/conv2d_25/bias*
_output_shapes
: *
dtype0
¦
'conv_net_9/batch_normalization_25/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'conv_net_9/batch_normalization_25/gamma

;conv_net_9/batch_normalization_25/gamma/Read/ReadVariableOpReadVariableOp'conv_net_9/batch_normalization_25/gamma*
_output_shapes
: *
dtype0
¤
&conv_net_9/batch_normalization_25/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&conv_net_9/batch_normalization_25/beta

:conv_net_9/batch_normalization_25/beta/Read/ReadVariableOpReadVariableOp&conv_net_9/batch_normalization_25/beta*
_output_shapes
: *
dtype0
²
-conv_net_9/batch_normalization_25/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-conv_net_9/batch_normalization_25/moving_mean
«
Aconv_net_9/batch_normalization_25/moving_mean/Read/ReadVariableOpReadVariableOp-conv_net_9/batch_normalization_25/moving_mean*
_output_shapes
: *
dtype0
º
1conv_net_9/batch_normalization_25/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31conv_net_9/batch_normalization_25/moving_variance
³
Econv_net_9/batch_normalization_25/moving_variance/Read/ReadVariableOpReadVariableOp1conv_net_9/batch_normalization_25/moving_variance*
_output_shapes
: *
dtype0

conv_net_9/conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*,
shared_nameconv_net_9/conv2d_26/kernel

/conv_net_9/conv2d_26/kernel/Read/ReadVariableOpReadVariableOpconv_net_9/conv2d_26/kernel*&
_output_shapes
: @*
dtype0

conv_net_9/conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameconv_net_9/conv2d_26/bias

-conv_net_9/conv2d_26/bias/Read/ReadVariableOpReadVariableOpconv_net_9/conv2d_26/bias*
_output_shapes
:@*
dtype0
¦
'conv_net_9/batch_normalization_26/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'conv_net_9/batch_normalization_26/gamma

;conv_net_9/batch_normalization_26/gamma/Read/ReadVariableOpReadVariableOp'conv_net_9/batch_normalization_26/gamma*
_output_shapes
:@*
dtype0
¤
&conv_net_9/batch_normalization_26/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&conv_net_9/batch_normalization_26/beta

:conv_net_9/batch_normalization_26/beta/Read/ReadVariableOpReadVariableOp&conv_net_9/batch_normalization_26/beta*
_output_shapes
:@*
dtype0
²
-conv_net_9/batch_normalization_26/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-conv_net_9/batch_normalization_26/moving_mean
«
Aconv_net_9/batch_normalization_26/moving_mean/Read/ReadVariableOpReadVariableOp-conv_net_9/batch_normalization_26/moving_mean*
_output_shapes
:@*
dtype0
º
1conv_net_9/batch_normalization_26/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*B
shared_name31conv_net_9/batch_normalization_26/moving_variance
³
Econv_net_9/batch_normalization_26/moving_variance/Read/ReadVariableOpReadVariableOp1conv_net_9/batch_normalization_26/moving_variance*
_output_shapes
:@*
dtype0

conv_net_9/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	À
**
shared_nameconv_net_9/dense_8/kernel

-conv_net_9/dense_8/kernel/Read/ReadVariableOpReadVariableOpconv_net_9/dense_8/kernel*
_output_shapes
:	À
*
dtype0

conv_net_9/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameconv_net_9/dense_8/bias

+conv_net_9/dense_8/bias/Read/ReadVariableOpReadVariableOpconv_net_9/dense_8/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
¹8
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ô7
valueê7Bç7 Bà7

	conv1
batch_norm1
max1
dropout1
	conv2
batch_norm2
max2
dropout2
		conv3

batch_norm3
dropout3
flatten
	dense
regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api

axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
trainable_variables
 	variables
!	keras_api
R
"regularization_losses
#trainable_variables
$	variables
%	keras_api
R
&regularization_losses
'trainable_variables
(	variables
)	keras_api
h

*kernel
+bias
,regularization_losses
-trainable_variables
.	variables
/	keras_api

0axis
	1gamma
2beta
3moving_mean
4moving_variance
5regularization_losses
6trainable_variables
7	variables
8	keras_api
R
9regularization_losses
:trainable_variables
;	variables
<	keras_api
R
=regularization_losses
>trainable_variables
?	variables
@	keras_api
h

Akernel
Bbias
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api

Gaxis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
R
Pregularization_losses
Qtrainable_variables
R	variables
S	keras_api
R
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
h

Xkernel
Ybias
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
 
f
0
1
2
3
*4
+5
16
27
A8
B9
H10
I11
X12
Y13

0
1
2
3
4
5
*6
+7
18
29
310
411
A12
B13
H14
I15
J16
K17
X18
Y19
­
^non_trainable_variables
_layer_metrics
regularization_losses

`layers
alayer_regularization_losses
bmetrics
trainable_variables
	variables
 
XV
VARIABLE_VALUEconv_net_9/conv2d_24/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv_net_9/conv2d_24/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
cnon_trainable_variables
dlayer_metrics
regularization_losses

elayers
flayer_regularization_losses
gmetrics
trainable_variables
	variables
 
ig
VARIABLE_VALUE'conv_net_9/batch_normalization_24/gamma,batch_norm1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE&conv_net_9/batch_normalization_24/beta+batch_norm1/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE-conv_net_9/batch_normalization_24/moving_mean2batch_norm1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE1conv_net_9/batch_normalization_24/moving_variance6batch_norm1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
2
3
­
hnon_trainable_variables
ilayer_metrics
regularization_losses

jlayers
klayer_regularization_losses
lmetrics
trainable_variables
 	variables
 
 
 
­
mnon_trainable_variables
nlayer_metrics
"regularization_losses

olayers
player_regularization_losses
qmetrics
#trainable_variables
$	variables
 
 
 
­
rnon_trainable_variables
slayer_metrics
&regularization_losses

tlayers
ulayer_regularization_losses
vmetrics
'trainable_variables
(	variables
XV
VARIABLE_VALUEconv_net_9/conv2d_25/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv_net_9/conv2d_25/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

*0
+1

*0
+1
­
wnon_trainable_variables
xlayer_metrics
,regularization_losses

ylayers
zlayer_regularization_losses
{metrics
-trainable_variables
.	variables
 
ig
VARIABLE_VALUE'conv_net_9/batch_normalization_25/gamma,batch_norm2/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE&conv_net_9/batch_normalization_25/beta+batch_norm2/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE-conv_net_9/batch_normalization_25/moving_mean2batch_norm2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE1conv_net_9/batch_normalization_25/moving_variance6batch_norm2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21
32
43
®
|non_trainable_variables
}layer_metrics
5regularization_losses

~layers
layer_regularization_losses
metrics
6trainable_variables
7	variables
 
 
 
²
non_trainable_variables
layer_metrics
9regularization_losses
layers
 layer_regularization_losses
metrics
:trainable_variables
;	variables
 
 
 
²
non_trainable_variables
layer_metrics
=regularization_losses
layers
 layer_regularization_losses
metrics
>trainable_variables
?	variables
XV
VARIABLE_VALUEconv_net_9/conv2d_26/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv_net_9/conv2d_26/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

A0
B1

A0
B1
²
non_trainable_variables
layer_metrics
Cregularization_losses
layers
 layer_regularization_losses
metrics
Dtrainable_variables
E	variables
 
ig
VARIABLE_VALUE'conv_net_9/batch_normalization_26/gamma,batch_norm3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE&conv_net_9/batch_normalization_26/beta+batch_norm3/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE-conv_net_9/batch_normalization_26/moving_mean2batch_norm3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE1conv_net_9/batch_normalization_26/moving_variance6batch_norm3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

H0
I1

H0
I1
J2
K3
²
non_trainable_variables
layer_metrics
Lregularization_losses
layers
 layer_regularization_losses
metrics
Mtrainable_variables
N	variables
 
 
 
²
non_trainable_variables
layer_metrics
Pregularization_losses
layers
 layer_regularization_losses
metrics
Qtrainable_variables
R	variables
 
 
 
²
non_trainable_variables
layer_metrics
Tregularization_losses
layers
 layer_regularization_losses
metrics
Utrainable_variables
V	variables
VT
VARIABLE_VALUEconv_net_9/dense_8/kernel'dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEconv_net_9/dense_8/bias%dense/bias/.ATTRIBUTES/VARIABLE_VALUE
 

X0
Y1

X0
Y1
²
non_trainable_variables
 layer_metrics
Zregularization_losses
¡layers
 ¢layer_regularization_losses
£metrics
[trainable_variables
\	variables
*
0
1
32
43
J4
K5
 
^
0
1
2
3
4
5
6
7
	8

9
10
11
12
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

30
41
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

J0
K1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

serving_default_input_1Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ  
×
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv_net_9/conv2d_24/kernelconv_net_9/conv2d_24/bias1conv_net_9/batch_normalization_24/moving_variance'conv_net_9/batch_normalization_24/gamma-conv_net_9/batch_normalization_24/moving_mean&conv_net_9/batch_normalization_24/betaconv_net_9/conv2d_25/kernelconv_net_9/conv2d_25/bias1conv_net_9/batch_normalization_25/moving_variance'conv_net_9/batch_normalization_25/gamma-conv_net_9/batch_normalization_25/moving_mean&conv_net_9/batch_normalization_25/betaconv_net_9/conv2d_26/kernelconv_net_9/conv2d_26/bias1conv_net_9/batch_normalization_26/moving_variance'conv_net_9/batch_normalization_26/gamma-conv_net_9/batch_normalization_26/moving_mean&conv_net_9/batch_normalization_26/betaconv_net_9/dense_8/kernelconv_net_9/dense_8/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *0
f+R)
'__inference_signature_wrapper_534954922
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¼
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/conv_net_9/conv2d_24/kernel/Read/ReadVariableOp-conv_net_9/conv2d_24/bias/Read/ReadVariableOp;conv_net_9/batch_normalization_24/gamma/Read/ReadVariableOp:conv_net_9/batch_normalization_24/beta/Read/ReadVariableOpAconv_net_9/batch_normalization_24/moving_mean/Read/ReadVariableOpEconv_net_9/batch_normalization_24/moving_variance/Read/ReadVariableOp/conv_net_9/conv2d_25/kernel/Read/ReadVariableOp-conv_net_9/conv2d_25/bias/Read/ReadVariableOp;conv_net_9/batch_normalization_25/gamma/Read/ReadVariableOp:conv_net_9/batch_normalization_25/beta/Read/ReadVariableOpAconv_net_9/batch_normalization_25/moving_mean/Read/ReadVariableOpEconv_net_9/batch_normalization_25/moving_variance/Read/ReadVariableOp/conv_net_9/conv2d_26/kernel/Read/ReadVariableOp-conv_net_9/conv2d_26/bias/Read/ReadVariableOp;conv_net_9/batch_normalization_26/gamma/Read/ReadVariableOp:conv_net_9/batch_normalization_26/beta/Read/ReadVariableOpAconv_net_9/batch_normalization_26/moving_mean/Read/ReadVariableOpEconv_net_9/batch_normalization_26/moving_variance/Read/ReadVariableOp-conv_net_9/dense_8/kernel/Read/ReadVariableOp+conv_net_9/dense_8/bias/Read/ReadVariableOpConst*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *+
f&R$
"__inference__traced_save_534956354
§
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_net_9/conv2d_24/kernelconv_net_9/conv2d_24/bias'conv_net_9/batch_normalization_24/gamma&conv_net_9/batch_normalization_24/beta-conv_net_9/batch_normalization_24/moving_mean1conv_net_9/batch_normalization_24/moving_varianceconv_net_9/conv2d_25/kernelconv_net_9/conv2d_25/bias'conv_net_9/batch_normalization_25/gamma&conv_net_9/batch_normalization_25/beta-conv_net_9/batch_normalization_25/moving_mean1conv_net_9/batch_normalization_25/moving_varianceconv_net_9/conv2d_26/kernelconv_net_9/conv2d_26/bias'conv_net_9/batch_normalization_26/gamma&conv_net_9/batch_normalization_26/beta-conv_net_9/batch_normalization_26/moving_mean1conv_net_9/batch_normalization_26/moving_varianceconv_net_9/dense_8/kernelconv_net_9/dense_8/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *.
f)R'
%__inference__traced_restore_534956424ä
è

+__inference_dense_8_layer_call_fn_534956271

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_8_layer_call_and_return_conditional_losses_5349545962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
Z
Á
%__inference__traced_restore_534956424
file_prefix0
,assignvariableop_conv_net_9_conv2d_24_kernel0
,assignvariableop_1_conv_net_9_conv2d_24_bias>
:assignvariableop_2_conv_net_9_batch_normalization_24_gamma=
9assignvariableop_3_conv_net_9_batch_normalization_24_betaD
@assignvariableop_4_conv_net_9_batch_normalization_24_moving_meanH
Dassignvariableop_5_conv_net_9_batch_normalization_24_moving_variance2
.assignvariableop_6_conv_net_9_conv2d_25_kernel0
,assignvariableop_7_conv_net_9_conv2d_25_bias>
:assignvariableop_8_conv_net_9_batch_normalization_25_gamma=
9assignvariableop_9_conv_net_9_batch_normalization_25_betaE
Aassignvariableop_10_conv_net_9_batch_normalization_25_moving_meanI
Eassignvariableop_11_conv_net_9_batch_normalization_25_moving_variance3
/assignvariableop_12_conv_net_9_conv2d_26_kernel1
-assignvariableop_13_conv_net_9_conv2d_26_bias?
;assignvariableop_14_conv_net_9_batch_normalization_26_gamma>
:assignvariableop_15_conv_net_9_batch_normalization_26_betaE
Aassignvariableop_16_conv_net_9_batch_normalization_26_moving_meanI
Eassignvariableop_17_conv_net_9_batch_normalization_26_moving_variance1
-assignvariableop_18_conv_net_9_dense_8_kernel/
+assignvariableop_19_conv_net_9_dense_8_bias
identity_21¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¼
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*È
value¾B»B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB,batch_norm1/gamma/.ATTRIBUTES/VARIABLE_VALUEB+batch_norm1/beta/.ATTRIBUTES/VARIABLE_VALUEB2batch_norm1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB,batch_norm2/gamma/.ATTRIBUTES/VARIABLE_VALUEB+batch_norm2/beta/.ATTRIBUTES/VARIABLE_VALUEB2batch_norm2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB,batch_norm3/gamma/.ATTRIBUTES/VARIABLE_VALUEB+batch_norm3/beta/.ATTRIBUTES/VARIABLE_VALUEB2batch_norm3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¸
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity«
AssignVariableOpAssignVariableOp,assignvariableop_conv_net_9_conv2d_24_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1±
AssignVariableOp_1AssignVariableOp,assignvariableop_1_conv_net_9_conv2d_24_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¿
AssignVariableOp_2AssignVariableOp:assignvariableop_2_conv_net_9_batch_normalization_24_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¾
AssignVariableOp_3AssignVariableOp9assignvariableop_3_conv_net_9_batch_normalization_24_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Å
AssignVariableOp_4AssignVariableOp@assignvariableop_4_conv_net_9_batch_normalization_24_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5É
AssignVariableOp_5AssignVariableOpDassignvariableop_5_conv_net_9_batch_normalization_24_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6³
AssignVariableOp_6AssignVariableOp.assignvariableop_6_conv_net_9_conv2d_25_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7±
AssignVariableOp_7AssignVariableOp,assignvariableop_7_conv_net_9_conv2d_25_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¿
AssignVariableOp_8AssignVariableOp:assignvariableop_8_conv_net_9_batch_normalization_25_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¾
AssignVariableOp_9AssignVariableOp9assignvariableop_9_conv_net_9_batch_normalization_25_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10É
AssignVariableOp_10AssignVariableOpAassignvariableop_10_conv_net_9_batch_normalization_25_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Í
AssignVariableOp_11AssignVariableOpEassignvariableop_11_conv_net_9_batch_normalization_25_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12·
AssignVariableOp_12AssignVariableOp/assignvariableop_12_conv_net_9_conv2d_26_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13µ
AssignVariableOp_13AssignVariableOp-assignvariableop_13_conv_net_9_conv2d_26_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ã
AssignVariableOp_14AssignVariableOp;assignvariableop_14_conv_net_9_batch_normalization_26_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Â
AssignVariableOp_15AssignVariableOp:assignvariableop_15_conv_net_9_batch_normalization_26_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16É
AssignVariableOp_16AssignVariableOpAassignvariableop_16_conv_net_9_batch_normalization_26_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Í
AssignVariableOp_17AssignVariableOpEassignvariableop_17_conv_net_9_batch_normalization_26_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18µ
AssignVariableOp_18AssignVariableOp-assignvariableop_18_conv_net_9_dense_8_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19³
AssignVariableOp_19AssignVariableOp+assignvariableop_19_conv_net_9_dense_8_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_199
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_20
Identity_21IdentityIdentity_20:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_21"#
identity_21Identity_21:output:0*e
_input_shapesT
R: ::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ë

.__inference_conv_net_9_layer_call_fn_534955214
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*0
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_conv_net_9_layer_call_and_return_conditional_losses_5349547302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_1
ì
g
I__inference_dropout_24_layer_call_and_return_conditional_losses_534954258

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ï
h
I__inference_dropout_25_layer_call_and_return_conditional_losses_534954404

inputs
identityg
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB 2      ô?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02&
$dropout/random_uniform/RandomUniformy
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB 2É?2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
³

U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_534954506

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul~
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
.
Ô
U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_534955869

inputs
assignmovingavg_534955842
assignmovingavg_1_534955849)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
: 2
moments/StopGradient¾
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"moments/variance/reduction_indicesº
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1¡
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/534955842*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay²
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*,
_class"
 loc:@AssignMovingAvg/534955842*
_output_shapes
: 2
AssignMovingAvg/Cast
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_534955842*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpÆ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/534955842*
_output_shapes
: 2
AssignMovingAvg/sub·
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*,
_class"
 loc:@AssignMovingAvg/534955842*
_output_shapes
: 2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_534955842AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/534955842*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/534955849*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decayº
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*.
_class$
" loc:@AssignMovingAvg_1/534955849*
_output_shapes
: 2
AssignMovingAvg_1/Cast
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_534955849*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpÐ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534955849*
_output_shapes
: 2
AssignMovingAvg_1/subÁ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534955849*
_output_shapes
: 2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_534955849AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/534955849*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1Ï
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¬
­
:__inference_batch_normalization_26_layer_call_fn_534956214

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *^
fYRW
U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_5349541052
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ò
®
F__inference_dense_8_layer_call_and_return_conditional_losses_534954596

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	À
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
Í
g
.__inference_dropout_26_layer_call_fn_534956236

inputs
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_26_layer_call_and_return_conditional_losses_5349545542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ì
g
I__inference_dropout_25_layer_call_and_return_conditional_losses_534954409

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¬
­
:__inference_batch_normalization_24_layer_call_fn_534955784

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *^
fYRW
U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_5349537932
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


-__inference_conv2d_25_layer_call_fn_534955831

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_conv2d_25_layer_call_and_return_conditional_losses_5349542822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¿

.__inference_conv_net_9_layer_call_fn_534955596
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_conv_net_9_layer_call_and_return_conditional_losses_5349548322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  

_user_specified_namex
Ô

$__inference__wrapped_model_534953660
input_17
3conv_net_9_conv2d_24_conv2d_readvariableop_resource8
4conv_net_9_conv2d_24_biasadd_readvariableop_resourceG
Cconv_net_9_batch_normalization_24_batchnorm_readvariableop_resourceK
Gconv_net_9_batch_normalization_24_batchnorm_mul_readvariableop_resourceI
Econv_net_9_batch_normalization_24_batchnorm_readvariableop_1_resourceI
Econv_net_9_batch_normalization_24_batchnorm_readvariableop_2_resource7
3conv_net_9_conv2d_25_conv2d_readvariableop_resource8
4conv_net_9_conv2d_25_biasadd_readvariableop_resourceG
Cconv_net_9_batch_normalization_25_batchnorm_readvariableop_resourceK
Gconv_net_9_batch_normalization_25_batchnorm_mul_readvariableop_resourceI
Econv_net_9_batch_normalization_25_batchnorm_readvariableop_1_resourceI
Econv_net_9_batch_normalization_25_batchnorm_readvariableop_2_resource7
3conv_net_9_conv2d_26_conv2d_readvariableop_resource8
4conv_net_9_conv2d_26_biasadd_readvariableop_resourceG
Cconv_net_9_batch_normalization_26_batchnorm_readvariableop_resourceK
Gconv_net_9_batch_normalization_26_batchnorm_mul_readvariableop_resourceI
Econv_net_9_batch_normalization_26_batchnorm_readvariableop_1_resourceI
Econv_net_9_batch_normalization_26_batchnorm_readvariableop_2_resource5
1conv_net_9_dense_8_matmul_readvariableop_resource6
2conv_net_9_dense_8_biasadd_readvariableop_resource
identityÔ
*conv_net_9/conv2d_24/Conv2D/ReadVariableOpReadVariableOp3conv_net_9_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*conv_net_9/conv2d_24/Conv2D/ReadVariableOpä
conv_net_9/conv2d_24/Conv2DConv2Dinput_12conv_net_9/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv_net_9/conv2d_24/Conv2DË
+conv_net_9/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp4conv_net_9_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+conv_net_9/conv2d_24/BiasAdd/ReadVariableOpÜ
conv_net_9/conv2d_24/BiasAddBiasAdd$conv_net_9/conv2d_24/Conv2D:output:03conv_net_9/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv_net_9/conv2d_24/BiasAdd
conv_net_9/conv2d_24/ReluRelu%conv_net_9/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv_net_9/conv2d_24/Reluø
:conv_net_9/batch_normalization_24/batchnorm/ReadVariableOpReadVariableOpCconv_net_9_batch_normalization_24_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02<
:conv_net_9/batch_normalization_24/batchnorm/ReadVariableOp¯
1conv_net_9/batch_normalization_24/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?23
1conv_net_9/batch_normalization_24/batchnorm/add/y
/conv_net_9/batch_normalization_24/batchnorm/addAddV2Bconv_net_9/batch_normalization_24/batchnorm/ReadVariableOp:value:0:conv_net_9/batch_normalization_24/batchnorm/add/y:output:0*
T0*
_output_shapes
: 21
/conv_net_9/batch_normalization_24/batchnorm/addÉ
1conv_net_9/batch_normalization_24/batchnorm/RsqrtRsqrt3conv_net_9/batch_normalization_24/batchnorm/add:z:0*
T0*
_output_shapes
: 23
1conv_net_9/batch_normalization_24/batchnorm/Rsqrt
>conv_net_9/batch_normalization_24/batchnorm/mul/ReadVariableOpReadVariableOpGconv_net_9_batch_normalization_24_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02@
>conv_net_9/batch_normalization_24/batchnorm/mul/ReadVariableOp
/conv_net_9/batch_normalization_24/batchnorm/mulMul5conv_net_9/batch_normalization_24/batchnorm/Rsqrt:y:0Fconv_net_9/batch_normalization_24/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 21
/conv_net_9/batch_normalization_24/batchnorm/mul
1conv_net_9/batch_normalization_24/batchnorm/mul_1Mul'conv_net_9/conv2d_24/Relu:activations:03conv_net_9/batch_normalization_24/batchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 23
1conv_net_9/batch_normalization_24/batchnorm/mul_1þ
<conv_net_9/batch_normalization_24/batchnorm/ReadVariableOp_1ReadVariableOpEconv_net_9_batch_normalization_24_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02>
<conv_net_9/batch_normalization_24/batchnorm/ReadVariableOp_1
1conv_net_9/batch_normalization_24/batchnorm/mul_2MulDconv_net_9/batch_normalization_24/batchnorm/ReadVariableOp_1:value:03conv_net_9/batch_normalization_24/batchnorm/mul:z:0*
T0*
_output_shapes
: 23
1conv_net_9/batch_normalization_24/batchnorm/mul_2þ
<conv_net_9/batch_normalization_24/batchnorm/ReadVariableOp_2ReadVariableOpEconv_net_9_batch_normalization_24_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02>
<conv_net_9/batch_normalization_24/batchnorm/ReadVariableOp_2
/conv_net_9/batch_normalization_24/batchnorm/subSubDconv_net_9/batch_normalization_24/batchnorm/ReadVariableOp_2:value:05conv_net_9/batch_normalization_24/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 21
/conv_net_9/batch_normalization_24/batchnorm/sub
1conv_net_9/batch_normalization_24/batchnorm/add_1AddV25conv_net_9/batch_normalization_24/batchnorm/mul_1:z:03conv_net_9/batch_normalization_24/batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 23
1conv_net_9/batch_normalization_24/batchnorm/add_1
#conv_net_9/max_pooling2d_16/MaxPoolMaxPool5conv_net_9/batch_normalization_24/batchnorm/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2%
#conv_net_9/max_pooling2d_16/MaxPool´
conv_net_9/dropout_24/IdentityIdentity,conv_net_9/max_pooling2d_16/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
conv_net_9/dropout_24/IdentityÔ
*conv_net_9/conv2d_25/Conv2D/ReadVariableOpReadVariableOp3conv_net_9_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*conv_net_9/conv2d_25/Conv2D/ReadVariableOp
conv_net_9/conv2d_25/Conv2DConv2D'conv_net_9/dropout_24/Identity:output:02conv_net_9/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *
paddingVALID*
strides
2
conv_net_9/conv2d_25/Conv2DË
+conv_net_9/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp4conv_net_9_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+conv_net_9/conv2d_25/BiasAdd/ReadVariableOpÜ
conv_net_9/conv2d_25/BiasAddBiasAdd$conv_net_9/conv2d_25/Conv2D:output:03conv_net_9/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
conv_net_9/conv2d_25/BiasAdd
conv_net_9/conv2d_25/ReluRelu%conv_net_9/conv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
conv_net_9/conv2d_25/Reluø
:conv_net_9/batch_normalization_25/batchnorm/ReadVariableOpReadVariableOpCconv_net_9_batch_normalization_25_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02<
:conv_net_9/batch_normalization_25/batchnorm/ReadVariableOp¯
1conv_net_9/batch_normalization_25/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?23
1conv_net_9/batch_normalization_25/batchnorm/add/y
/conv_net_9/batch_normalization_25/batchnorm/addAddV2Bconv_net_9/batch_normalization_25/batchnorm/ReadVariableOp:value:0:conv_net_9/batch_normalization_25/batchnorm/add/y:output:0*
T0*
_output_shapes
: 21
/conv_net_9/batch_normalization_25/batchnorm/addÉ
1conv_net_9/batch_normalization_25/batchnorm/RsqrtRsqrt3conv_net_9/batch_normalization_25/batchnorm/add:z:0*
T0*
_output_shapes
: 23
1conv_net_9/batch_normalization_25/batchnorm/Rsqrt
>conv_net_9/batch_normalization_25/batchnorm/mul/ReadVariableOpReadVariableOpGconv_net_9_batch_normalization_25_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02@
>conv_net_9/batch_normalization_25/batchnorm/mul/ReadVariableOp
/conv_net_9/batch_normalization_25/batchnorm/mulMul5conv_net_9/batch_normalization_25/batchnorm/Rsqrt:y:0Fconv_net_9/batch_normalization_25/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 21
/conv_net_9/batch_normalization_25/batchnorm/mul
1conv_net_9/batch_normalization_25/batchnorm/mul_1Mul'conv_net_9/conv2d_25/Relu:activations:03conv_net_9/batch_normalization_25/batchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 23
1conv_net_9/batch_normalization_25/batchnorm/mul_1þ
<conv_net_9/batch_normalization_25/batchnorm/ReadVariableOp_1ReadVariableOpEconv_net_9_batch_normalization_25_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02>
<conv_net_9/batch_normalization_25/batchnorm/ReadVariableOp_1
1conv_net_9/batch_normalization_25/batchnorm/mul_2MulDconv_net_9/batch_normalization_25/batchnorm/ReadVariableOp_1:value:03conv_net_9/batch_normalization_25/batchnorm/mul:z:0*
T0*
_output_shapes
: 23
1conv_net_9/batch_normalization_25/batchnorm/mul_2þ
<conv_net_9/batch_normalization_25/batchnorm/ReadVariableOp_2ReadVariableOpEconv_net_9_batch_normalization_25_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02>
<conv_net_9/batch_normalization_25/batchnorm/ReadVariableOp_2
/conv_net_9/batch_normalization_25/batchnorm/subSubDconv_net_9/batch_normalization_25/batchnorm/ReadVariableOp_2:value:05conv_net_9/batch_normalization_25/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 21
/conv_net_9/batch_normalization_25/batchnorm/sub
1conv_net_9/batch_normalization_25/batchnorm/add_1AddV25conv_net_9/batch_normalization_25/batchnorm/mul_1:z:03conv_net_9/batch_normalization_25/batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 23
1conv_net_9/batch_normalization_25/batchnorm/add_1
#conv_net_9/max_pooling2d_17/MaxPoolMaxPool5conv_net_9/batch_normalization_25/batchnorm/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2%
#conv_net_9/max_pooling2d_17/MaxPool´
conv_net_9/dropout_25/IdentityIdentity,conv_net_9/max_pooling2d_17/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
conv_net_9/dropout_25/IdentityÔ
*conv_net_9/conv2d_26/Conv2D/ReadVariableOpReadVariableOp3conv_net_9_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*conv_net_9/conv2d_26/Conv2D/ReadVariableOp
conv_net_9/conv2d_26/Conv2DConv2D'conv_net_9/dropout_25/Identity:output:02conv_net_9/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv_net_9/conv2d_26/Conv2DË
+conv_net_9/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp4conv_net_9_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+conv_net_9/conv2d_26/BiasAdd/ReadVariableOpÜ
conv_net_9/conv2d_26/BiasAddBiasAdd$conv_net_9/conv2d_26/Conv2D:output:03conv_net_9/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv_net_9/conv2d_26/BiasAdd
conv_net_9/conv2d_26/ReluRelu%conv_net_9/conv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv_net_9/conv2d_26/Reluø
:conv_net_9/batch_normalization_26/batchnorm/ReadVariableOpReadVariableOpCconv_net_9_batch_normalization_26_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02<
:conv_net_9/batch_normalization_26/batchnorm/ReadVariableOp¯
1conv_net_9/batch_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?23
1conv_net_9/batch_normalization_26/batchnorm/add/y
/conv_net_9/batch_normalization_26/batchnorm/addAddV2Bconv_net_9/batch_normalization_26/batchnorm/ReadVariableOp:value:0:conv_net_9/batch_normalization_26/batchnorm/add/y:output:0*
T0*
_output_shapes
:@21
/conv_net_9/batch_normalization_26/batchnorm/addÉ
1conv_net_9/batch_normalization_26/batchnorm/RsqrtRsqrt3conv_net_9/batch_normalization_26/batchnorm/add:z:0*
T0*
_output_shapes
:@23
1conv_net_9/batch_normalization_26/batchnorm/Rsqrt
>conv_net_9/batch_normalization_26/batchnorm/mul/ReadVariableOpReadVariableOpGconv_net_9_batch_normalization_26_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02@
>conv_net_9/batch_normalization_26/batchnorm/mul/ReadVariableOp
/conv_net_9/batch_normalization_26/batchnorm/mulMul5conv_net_9/batch_normalization_26/batchnorm/Rsqrt:y:0Fconv_net_9/batch_normalization_26/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@21
/conv_net_9/batch_normalization_26/batchnorm/mul
1conv_net_9/batch_normalization_26/batchnorm/mul_1Mul'conv_net_9/conv2d_26/Relu:activations:03conv_net_9/batch_normalization_26/batchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@23
1conv_net_9/batch_normalization_26/batchnorm/mul_1þ
<conv_net_9/batch_normalization_26/batchnorm/ReadVariableOp_1ReadVariableOpEconv_net_9_batch_normalization_26_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02>
<conv_net_9/batch_normalization_26/batchnorm/ReadVariableOp_1
1conv_net_9/batch_normalization_26/batchnorm/mul_2MulDconv_net_9/batch_normalization_26/batchnorm/ReadVariableOp_1:value:03conv_net_9/batch_normalization_26/batchnorm/mul:z:0*
T0*
_output_shapes
:@23
1conv_net_9/batch_normalization_26/batchnorm/mul_2þ
<conv_net_9/batch_normalization_26/batchnorm/ReadVariableOp_2ReadVariableOpEconv_net_9_batch_normalization_26_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02>
<conv_net_9/batch_normalization_26/batchnorm/ReadVariableOp_2
/conv_net_9/batch_normalization_26/batchnorm/subSubDconv_net_9/batch_normalization_26/batchnorm/ReadVariableOp_2:value:05conv_net_9/batch_normalization_26/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@21
/conv_net_9/batch_normalization_26/batchnorm/sub
1conv_net_9/batch_normalization_26/batchnorm/add_1AddV25conv_net_9/batch_normalization_26/batchnorm/mul_1:z:03conv_net_9/batch_normalization_26/batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@23
1conv_net_9/batch_normalization_26/batchnorm/add_1½
conv_net_9/dropout_26/IdentityIdentity5conv_net_9/batch_normalization_26/batchnorm/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
conv_net_9/dropout_26/Identity
conv_net_9/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
conv_net_9/flatten_8/ConstÈ
conv_net_9/flatten_8/ReshapeReshape'conv_net_9/dropout_26/Identity:output:0#conv_net_9/flatten_8/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
conv_net_9/flatten_8/ReshapeÇ
(conv_net_9/dense_8/MatMul/ReadVariableOpReadVariableOp1conv_net_9_dense_8_matmul_readvariableop_resource*
_output_shapes
:	À
*
dtype02*
(conv_net_9/dense_8/MatMul/ReadVariableOpË
conv_net_9/dense_8/MatMulMatMul%conv_net_9/flatten_8/Reshape:output:00conv_net_9/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
conv_net_9/dense_8/MatMulÅ
)conv_net_9/dense_8/BiasAdd/ReadVariableOpReadVariableOp2conv_net_9_dense_8_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02+
)conv_net_9/dense_8/BiasAdd/ReadVariableOpÍ
conv_net_9/dense_8/BiasAddBiasAdd#conv_net_9/dense_8/MatMul:product:01conv_net_9/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
conv_net_9/dense_8/BiasAddw
IdentityIdentity#conv_net_9/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ  :::::::::::::::::::::X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_1
F
ÿ
I__inference_conv_net_9_layer_call_and_return_conditional_losses_534954730
x
conv2d_24_534954676
conv2d_24_534954678$
 batch_normalization_24_534954681$
 batch_normalization_24_534954683$
 batch_normalization_24_534954685$
 batch_normalization_24_534954687
conv2d_25_534954692
conv2d_25_534954694$
 batch_normalization_25_534954697$
 batch_normalization_25_534954699$
 batch_normalization_25_534954701$
 batch_normalization_25_534954703
conv2d_26_534954708
conv2d_26_534954710$
 batch_normalization_26_534954713$
 batch_normalization_26_534954715$
 batch_normalization_26_534954717$
 batch_normalization_26_534954719
dense_8_534954724
dense_8_534954726
identity¢.batch_normalization_24/StatefulPartitionedCall¢.batch_normalization_25/StatefulPartitionedCall¢.batch_normalization_26/StatefulPartitionedCall¢!conv2d_24/StatefulPartitionedCall¢!conv2d_25/StatefulPartitionedCall¢!conv2d_26/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢"dropout_24/StatefulPartitionedCall¢"dropout_25/StatefulPartitionedCall¢"dropout_26/StatefulPartitionedCallª
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCallxconv2d_24_534954676conv2d_24_534954678*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_conv2d_24_layer_call_and_return_conditional_losses_5349541312#
!conv2d_24/StatefulPartitionedCallÚ
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0 batch_normalization_24_534954681 batch_normalization_24_534954683 batch_normalization_24_534954685 batch_normalization_24_534954687*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *^
fYRW
U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_53495418420
.batch_normalization_24/StatefulPartitionedCall­
 max_pooling2d_16/PartitionedCallPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_5349538102"
 max_pooling2d_16/PartitionedCall¥
"dropout_24/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_24_layer_call_and_return_conditional_losses_5349542532$
"dropout_24/StatefulPartitionedCallÔ
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall+dropout_24/StatefulPartitionedCall:output:0conv2d_25_534954692conv2d_25_534954694*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_conv2d_25_layer_call_and_return_conditional_losses_5349542822#
!conv2d_25/StatefulPartitionedCallÚ
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0 batch_normalization_25_534954697 batch_normalization_25_534954699 batch_normalization_25_534954701 batch_normalization_25_534954703*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *^
fYRW
U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_53495433520
.batch_normalization_25/StatefulPartitionedCall­
 max_pooling2d_17/PartitionedCallPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_5349539662"
 max_pooling2d_17/PartitionedCallÊ
"dropout_25/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_17/PartitionedCall:output:0#^dropout_24/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_25_layer_call_and_return_conditional_losses_5349544042$
"dropout_25/StatefulPartitionedCallÔ
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_25/StatefulPartitionedCall:output:0conv2d_26_534954708conv2d_26_534954710*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_conv2d_26_layer_call_and_return_conditional_losses_5349544332#
!conv2d_26/StatefulPartitionedCallÚ
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0 batch_normalization_26_534954713 batch_normalization_26_534954715 batch_normalization_26_534954717 batch_normalization_26_534954719*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *^
fYRW
U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_53495448620
.batch_normalization_26/StatefulPartitionedCallØ
"dropout_26/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0#^dropout_25/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_26_layer_call_and_return_conditional_losses_5349545542$
"dropout_26/StatefulPartitionedCall
flatten_8/PartitionedCallPartitionedCall+dropout_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_flatten_8_layer_call_and_return_conditional_losses_5349545782
flatten_8/PartitionedCall¹
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_8_534954724dense_8_534954726*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_8_layer_call_and_return_conditional_losses_5349545962!
dense_8/StatefulPartitionedCall
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_26/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall#^dropout_24/StatefulPartitionedCall#^dropout_25/StatefulPartitionedCall#^dropout_26/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2H
"dropout_24/StatefulPartitionedCall"dropout_24/StatefulPartitionedCall2H
"dropout_25/StatefulPartitionedCall"dropout_25/StatefulPartitionedCall2H
"dropout_26/StatefulPartitionedCall"dropout_26/StatefulPartitionedCall:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  

_user_specified_namex
-
Ô
U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_534954184

inputs
assignmovingavg_534954157
assignmovingavg_1_534954164)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
: 2
moments/StopGradient¬
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"moments/variance/reduction_indicesº
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1¡
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/534954157*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay²
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*,
_class"
 loc:@AssignMovingAvg/534954157*
_output_shapes
: 2
AssignMovingAvg/Cast
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_534954157*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpÆ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/534954157*
_output_shapes
: 2
AssignMovingAvg/sub·
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*,
_class"
 loc:@AssignMovingAvg/534954157*
_output_shapes
: 2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_534954157AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/534954157*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/534954164*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decayº
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*.
_class$
" loc:@AssignMovingAvg_1/534954164*
_output_shapes
: 2
AssignMovingAvg_1/Cast
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_534954164*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpÐ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534954164*
_output_shapes
: 2
AssignMovingAvg_1/subÁ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534954164*
_output_shapes
: 2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_534954164AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/534954164*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul~
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1½
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ï
h
I__inference_dropout_26_layer_call_and_return_conditional_losses_534954554

inputs
identityg
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB 2·mÛ¶mÛö?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
$dropout/random_uniform/RandomUniformy
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB 2433333Ó?2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
³

U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_534955674

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul~
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ :::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
.
Ô
U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_534953760

inputs
assignmovingavg_534953733
assignmovingavg_1_534953740)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
: 2
moments/StopGradient¾
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"moments/variance/reduction_indicesº
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1¡
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/534953733*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay²
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*,
_class"
 loc:@AssignMovingAvg/534953733*
_output_shapes
: 2
AssignMovingAvg/Cast
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_534953733*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpÆ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/534953733*
_output_shapes
: 2
AssignMovingAvg/sub·
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*,
_class"
 loc:@AssignMovingAvg/534953733*
_output_shapes
: 2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_534953733AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/534953733*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/534953740*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decayº
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*.
_class$
" loc:@AssignMovingAvg_1/534953740*
_output_shapes
: 2
AssignMovingAvg_1/Cast
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_534953740*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpÐ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534953740*
_output_shapes
: 2
AssignMovingAvg_1/subÁ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534953740*
_output_shapes
: 2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_534953740AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/534953740*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1Ï
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ï
h
I__inference_dropout_26_layer_call_and_return_conditional_losses_534956226

inputs
identityg
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB 2·mÛ¶mÛö?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
$dropout/random_uniform/RandomUniformy
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB 2433333Ó?2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
°
H__inference_conv2d_26_layer_call_and_return_conditional_losses_534956037

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

k
O__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_534953810

inputs
identity¶
MaxPoolMaxPoolinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
°
H__inference_conv2d_26_layer_call_and_return_conditional_losses_534954433

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
-
Ô
U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_534955953

inputs
assignmovingavg_534955926
assignmovingavg_1_534955933)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
: 2
moments/StopGradient¬
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"moments/variance/reduction_indicesº
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1¡
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/534955926*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay²
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*,
_class"
 loc:@AssignMovingAvg/534955926*
_output_shapes
: 2
AssignMovingAvg/Cast
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_534955926*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpÆ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/534955926*
_output_shapes
: 2
AssignMovingAvg/sub·
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*,
_class"
 loc:@AssignMovingAvg/534955926*
_output_shapes
: 2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_534955926AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/534955926*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/534955933*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decayº
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*.
_class$
" loc:@AssignMovingAvg_1/534955933*
_output_shapes
: 2
AssignMovingAvg_1/Cast
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_534955933*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpÐ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534955933*
_output_shapes
: 2
AssignMovingAvg_1/subÁ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534955933*
_output_shapes
: 2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_534955933AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/534955933*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul~
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
batchnorm/add_1½
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ

 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
 
_user_specified_nameinputs


U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_534954105

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
³

U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_534955973

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul~
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ

 :::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
 
_user_specified_nameinputs
Á
J
.__inference_dropout_25_layer_call_fn_534956026

inputs
identityÔ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_25_layer_call_and_return_conditional_losses_5349544092
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_534955758

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
º
P
4__inference_max_pooling2d_17_layer_call_fn_534953972

inputs
identityõ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_5349539662
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
°
H__inference_conv2d_24_layer_call_and_return_conditional_losses_534954131

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ  :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ï
h
I__inference_dropout_24_layer_call_and_return_conditional_losses_534954253

inputs
identityg
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB 2rÇqÇñ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02&
$dropout/random_uniform/RandomUniformy
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB 2¹?2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
-
Ô
U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_534954486

inputs
assignmovingavg_534954459
assignmovingavg_1_534954466)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:@2
moments/StopGradient¬
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"moments/variance/reduction_indicesº
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1¡
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/534954459*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay²
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*,
_class"
 loc:@AssignMovingAvg/534954459*
_output_shapes
: 2
AssignMovingAvg/Cast
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_534954459*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpÆ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/534954459*
_output_shapes
:@2
AssignMovingAvg/sub·
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*,
_class"
 loc:@AssignMovingAvg/534954459*
_output_shapes
:@2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_534954459AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/534954459*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/534954466*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decayº
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*.
_class$
" loc:@AssignMovingAvg_1/534954466*
_output_shapes
: 2
AssignMovingAvg_1/Cast
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_534954466*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÐ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534954466*
_output_shapes
:@2
AssignMovingAvg_1/subÁ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534954466*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_534954466AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/534954466*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul~
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
batchnorm/add_1½
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ú

I__inference_conv_net_9_layer_call_and_return_conditional_losses_534955420
x,
(conv2d_24_conv2d_readvariableop_resource-
)conv2d_24_biasadd_readvariableop_resource4
0batch_normalization_24_assignmovingavg_5349552776
2batch_normalization_24_assignmovingavg_1_534955284@
<batch_normalization_24_batchnorm_mul_readvariableop_resource<
8batch_normalization_24_batchnorm_readvariableop_resource,
(conv2d_25_conv2d_readvariableop_resource-
)conv2d_25_biasadd_readvariableop_resource4
0batch_normalization_25_assignmovingavg_5349553276
2batch_normalization_25_assignmovingavg_1_534955334@
<batch_normalization_25_batchnorm_mul_readvariableop_resource<
8batch_normalization_25_batchnorm_readvariableop_resource,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource4
0batch_normalization_26_assignmovingavg_5349553776
2batch_normalization_26_assignmovingavg_1_534955384@
<batch_normalization_26_batchnorm_mul_readvariableop_resource<
8batch_normalization_26_batchnorm_readvariableop_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource
identity¢:batch_normalization_24/AssignMovingAvg/AssignSubVariableOp¢<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp¢:batch_normalization_25/AssignMovingAvg/AssignSubVariableOp¢<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp¢:batch_normalization_26/AssignMovingAvg/AssignSubVariableOp¢<batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOp³
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_24/Conv2D/ReadVariableOp½
conv2d_24/Conv2DConv2Dx'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_24/Conv2Dª
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp°
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_24/BiasAdd~
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_24/ReluÃ
5batch_normalization_24/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          27
5batch_normalization_24/moments/mean/reduction_indicesò
#batch_normalization_24/moments/meanMeanconv2d_24/Relu:activations:0>batch_normalization_24/moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2%
#batch_normalization_24/moments/meanÉ
+batch_normalization_24/moments/StopGradientStopGradient,batch_normalization_24/moments/mean:output:0*
T0*&
_output_shapes
: 2-
+batch_normalization_24/moments/StopGradient
0batch_normalization_24/moments/SquaredDifferenceSquaredDifferenceconv2d_24/Relu:activations:04batch_normalization_24/moments/StopGradient:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0batch_normalization_24/moments/SquaredDifferenceË
9batch_normalization_24/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2;
9batch_normalization_24/moments/variance/reduction_indices
'batch_normalization_24/moments/varianceMean4batch_normalization_24/moments/SquaredDifference:z:0Bbatch_normalization_24/moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2)
'batch_normalization_24/moments/varianceÇ
&batch_normalization_24/moments/SqueezeSqueeze,batch_normalization_24/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2(
&batch_normalization_24/moments/SqueezeÏ
(batch_normalization_24/moments/Squeeze_1Squeeze0batch_normalization_24/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2*
(batch_normalization_24/moments/Squeeze_1æ
,batch_normalization_24/AssignMovingAvg/decayConst*C
_class9
75loc:@batch_normalization_24/AssignMovingAvg/534955277*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_24/AssignMovingAvg/decay
+batch_normalization_24/AssignMovingAvg/CastCast5batch_normalization_24/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*C
_class9
75loc:@batch_normalization_24/AssignMovingAvg/534955277*
_output_shapes
: 2-
+batch_normalization_24/AssignMovingAvg/CastÛ
5batch_normalization_24/AssignMovingAvg/ReadVariableOpReadVariableOp0batch_normalization_24_assignmovingavg_534955277*
_output_shapes
: *
dtype027
5batch_normalization_24/AssignMovingAvg/ReadVariableOp¹
*batch_normalization_24/AssignMovingAvg/subSub=batch_normalization_24/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_24/moments/Squeeze:output:0*
T0*C
_class9
75loc:@batch_normalization_24/AssignMovingAvg/534955277*
_output_shapes
: 2,
*batch_normalization_24/AssignMovingAvg/subª
*batch_normalization_24/AssignMovingAvg/mulMul.batch_normalization_24/AssignMovingAvg/sub:z:0/batch_normalization_24/AssignMovingAvg/Cast:y:0*
T0*C
_class9
75loc:@batch_normalization_24/AssignMovingAvg/534955277*
_output_shapes
: 2,
*batch_normalization_24/AssignMovingAvg/mul
:batch_normalization_24/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp0batch_normalization_24_assignmovingavg_534955277.batch_normalization_24/AssignMovingAvg/mul:z:06^batch_normalization_24/AssignMovingAvg/ReadVariableOp*C
_class9
75loc:@batch_normalization_24/AssignMovingAvg/534955277*
_output_shapes
 *
dtype02<
:batch_normalization_24/AssignMovingAvg/AssignSubVariableOpì
.batch_normalization_24/AssignMovingAvg_1/decayConst*E
_class;
97loc:@batch_normalization_24/AssignMovingAvg_1/534955284*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_24/AssignMovingAvg_1/decay
-batch_normalization_24/AssignMovingAvg_1/CastCast7batch_normalization_24/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*E
_class;
97loc:@batch_normalization_24/AssignMovingAvg_1/534955284*
_output_shapes
: 2/
-batch_normalization_24/AssignMovingAvg_1/Castá
7batch_normalization_24/AssignMovingAvg_1/ReadVariableOpReadVariableOp2batch_normalization_24_assignmovingavg_1_534955284*
_output_shapes
: *
dtype029
7batch_normalization_24/AssignMovingAvg_1/ReadVariableOpÃ
,batch_normalization_24/AssignMovingAvg_1/subSub?batch_normalization_24/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_24/moments/Squeeze_1:output:0*
T0*E
_class;
97loc:@batch_normalization_24/AssignMovingAvg_1/534955284*
_output_shapes
: 2.
,batch_normalization_24/AssignMovingAvg_1/sub´
,batch_normalization_24/AssignMovingAvg_1/mulMul0batch_normalization_24/AssignMovingAvg_1/sub:z:01batch_normalization_24/AssignMovingAvg_1/Cast:y:0*
T0*E
_class;
97loc:@batch_normalization_24/AssignMovingAvg_1/534955284*
_output_shapes
: 2.
,batch_normalization_24/AssignMovingAvg_1/mul
<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp2batch_normalization_24_assignmovingavg_1_5349552840batch_normalization_24/AssignMovingAvg_1/mul:z:08^batch_normalization_24/AssignMovingAvg_1/ReadVariableOp*E
_class;
97loc:@batch_normalization_24/AssignMovingAvg_1/534955284*
_output_shapes
 *
dtype02>
<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_24/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2(
&batch_normalization_24/batchnorm/add/yÞ
$batch_normalization_24/batchnorm/addAddV21batch_normalization_24/moments/Squeeze_1:output:0/batch_normalization_24/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_24/batchnorm/add¨
&batch_normalization_24/batchnorm/RsqrtRsqrt(batch_normalization_24/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_24/batchnorm/Rsqrtã
3batch_normalization_24/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_24_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_24/batchnorm/mul/ReadVariableOpá
$batch_normalization_24/batchnorm/mulMul*batch_normalization_24/batchnorm/Rsqrt:y:0;batch_normalization_24/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_24/batchnorm/mulÙ
&batch_normalization_24/batchnorm/mul_1Mulconv2d_24/Relu:activations:0(batch_normalization_24/batchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&batch_normalization_24/batchnorm/mul_1×
&batch_normalization_24/batchnorm/mul_2Mul/batch_normalization_24/moments/Squeeze:output:0(batch_normalization_24/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_24/batchnorm/mul_2×
/batch_normalization_24/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_24_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_24/batchnorm/ReadVariableOpÝ
$batch_normalization_24/batchnorm/subSub7batch_normalization_24/batchnorm/ReadVariableOp:value:0*batch_normalization_24/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_24/batchnorm/subé
&batch_normalization_24/batchnorm/add_1AddV2*batch_normalization_24/batchnorm/mul_1:z:0(batch_normalization_24/batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&batch_normalization_24/batchnorm/add_1á
max_pooling2d_16/MaxPoolMaxPool*batch_normalization_24/batchnorm/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_16/MaxPool}
dropout_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB 2rÇqÇñ?2
dropout_24/dropout/Const·
dropout_24/dropout/MulMul!max_pooling2d_16/MaxPool:output:0!dropout_24/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_24/dropout/Mul
dropout_24/dropout/ShapeShape!max_pooling2d_16/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_24/dropout/ShapeÝ
/dropout_24/dropout/random_uniform/RandomUniformRandomUniform!dropout_24/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype021
/dropout_24/dropout/random_uniform/RandomUniform
!dropout_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB 2¹?2#
!dropout_24/dropout/GreaterEqual/yò
dropout_24/dropout/GreaterEqualGreaterEqual8dropout_24/dropout/random_uniform/RandomUniform:output:0*dropout_24/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
dropout_24/dropout/GreaterEqual¨
dropout_24/dropout/CastCast#dropout_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_24/dropout/Cast®
dropout_24/dropout/Mul_1Muldropout_24/dropout/Mul:z:0dropout_24/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_24/dropout/Mul_1³
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_25/Conv2D/ReadVariableOpØ
conv2d_25/Conv2DConv2Ddropout_24/dropout/Mul_1:z:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *
paddingVALID*
strides
2
conv2d_25/Conv2Dª
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_25/BiasAdd/ReadVariableOp°
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
conv2d_25/BiasAdd~
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
conv2d_25/ReluÃ
5batch_normalization_25/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          27
5batch_normalization_25/moments/mean/reduction_indicesò
#batch_normalization_25/moments/meanMeanconv2d_25/Relu:activations:0>batch_normalization_25/moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2%
#batch_normalization_25/moments/meanÉ
+batch_normalization_25/moments/StopGradientStopGradient,batch_normalization_25/moments/mean:output:0*
T0*&
_output_shapes
: 2-
+batch_normalization_25/moments/StopGradient
0batch_normalization_25/moments/SquaredDifferenceSquaredDifferenceconv2d_25/Relu:activations:04batch_normalization_25/moments/StopGradient:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 22
0batch_normalization_25/moments/SquaredDifferenceË
9batch_normalization_25/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2;
9batch_normalization_25/moments/variance/reduction_indices
'batch_normalization_25/moments/varianceMean4batch_normalization_25/moments/SquaredDifference:z:0Bbatch_normalization_25/moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2)
'batch_normalization_25/moments/varianceÇ
&batch_normalization_25/moments/SqueezeSqueeze,batch_normalization_25/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2(
&batch_normalization_25/moments/SqueezeÏ
(batch_normalization_25/moments/Squeeze_1Squeeze0batch_normalization_25/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2*
(batch_normalization_25/moments/Squeeze_1æ
,batch_normalization_25/AssignMovingAvg/decayConst*C
_class9
75loc:@batch_normalization_25/AssignMovingAvg/534955327*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_25/AssignMovingAvg/decay
+batch_normalization_25/AssignMovingAvg/CastCast5batch_normalization_25/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*C
_class9
75loc:@batch_normalization_25/AssignMovingAvg/534955327*
_output_shapes
: 2-
+batch_normalization_25/AssignMovingAvg/CastÛ
5batch_normalization_25/AssignMovingAvg/ReadVariableOpReadVariableOp0batch_normalization_25_assignmovingavg_534955327*
_output_shapes
: *
dtype027
5batch_normalization_25/AssignMovingAvg/ReadVariableOp¹
*batch_normalization_25/AssignMovingAvg/subSub=batch_normalization_25/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_25/moments/Squeeze:output:0*
T0*C
_class9
75loc:@batch_normalization_25/AssignMovingAvg/534955327*
_output_shapes
: 2,
*batch_normalization_25/AssignMovingAvg/subª
*batch_normalization_25/AssignMovingAvg/mulMul.batch_normalization_25/AssignMovingAvg/sub:z:0/batch_normalization_25/AssignMovingAvg/Cast:y:0*
T0*C
_class9
75loc:@batch_normalization_25/AssignMovingAvg/534955327*
_output_shapes
: 2,
*batch_normalization_25/AssignMovingAvg/mul
:batch_normalization_25/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp0batch_normalization_25_assignmovingavg_534955327.batch_normalization_25/AssignMovingAvg/mul:z:06^batch_normalization_25/AssignMovingAvg/ReadVariableOp*C
_class9
75loc:@batch_normalization_25/AssignMovingAvg/534955327*
_output_shapes
 *
dtype02<
:batch_normalization_25/AssignMovingAvg/AssignSubVariableOpì
.batch_normalization_25/AssignMovingAvg_1/decayConst*E
_class;
97loc:@batch_normalization_25/AssignMovingAvg_1/534955334*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_25/AssignMovingAvg_1/decay
-batch_normalization_25/AssignMovingAvg_1/CastCast7batch_normalization_25/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*E
_class;
97loc:@batch_normalization_25/AssignMovingAvg_1/534955334*
_output_shapes
: 2/
-batch_normalization_25/AssignMovingAvg_1/Castá
7batch_normalization_25/AssignMovingAvg_1/ReadVariableOpReadVariableOp2batch_normalization_25_assignmovingavg_1_534955334*
_output_shapes
: *
dtype029
7batch_normalization_25/AssignMovingAvg_1/ReadVariableOpÃ
,batch_normalization_25/AssignMovingAvg_1/subSub?batch_normalization_25/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_25/moments/Squeeze_1:output:0*
T0*E
_class;
97loc:@batch_normalization_25/AssignMovingAvg_1/534955334*
_output_shapes
: 2.
,batch_normalization_25/AssignMovingAvg_1/sub´
,batch_normalization_25/AssignMovingAvg_1/mulMul0batch_normalization_25/AssignMovingAvg_1/sub:z:01batch_normalization_25/AssignMovingAvg_1/Cast:y:0*
T0*E
_class;
97loc:@batch_normalization_25/AssignMovingAvg_1/534955334*
_output_shapes
: 2.
,batch_normalization_25/AssignMovingAvg_1/mul
<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp2batch_normalization_25_assignmovingavg_1_5349553340batch_normalization_25/AssignMovingAvg_1/mul:z:08^batch_normalization_25/AssignMovingAvg_1/ReadVariableOp*E
_class;
97loc:@batch_normalization_25/AssignMovingAvg_1/534955334*
_output_shapes
 *
dtype02>
<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_25/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2(
&batch_normalization_25/batchnorm/add/yÞ
$batch_normalization_25/batchnorm/addAddV21batch_normalization_25/moments/Squeeze_1:output:0/batch_normalization_25/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_25/batchnorm/add¨
&batch_normalization_25/batchnorm/RsqrtRsqrt(batch_normalization_25/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_25/batchnorm/Rsqrtã
3batch_normalization_25/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_25_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_25/batchnorm/mul/ReadVariableOpá
$batch_normalization_25/batchnorm/mulMul*batch_normalization_25/batchnorm/Rsqrt:y:0;batch_normalization_25/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_25/batchnorm/mulÙ
&batch_normalization_25/batchnorm/mul_1Mulconv2d_25/Relu:activations:0(batch_normalization_25/batchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2(
&batch_normalization_25/batchnorm/mul_1×
&batch_normalization_25/batchnorm/mul_2Mul/batch_normalization_25/moments/Squeeze:output:0(batch_normalization_25/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_25/batchnorm/mul_2×
/batch_normalization_25/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_25_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_25/batchnorm/ReadVariableOpÝ
$batch_normalization_25/batchnorm/subSub7batch_normalization_25/batchnorm/ReadVariableOp:value:0*batch_normalization_25/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_25/batchnorm/subé
&batch_normalization_25/batchnorm/add_1AddV2*batch_normalization_25/batchnorm/mul_1:z:0(batch_normalization_25/batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2(
&batch_normalization_25/batchnorm/add_1á
max_pooling2d_17/MaxPoolMaxPool*batch_normalization_25/batchnorm/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_17/MaxPool}
dropout_25/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB 2      ô?2
dropout_25/dropout/Const·
dropout_25/dropout/MulMul!max_pooling2d_17/MaxPool:output:0!dropout_25/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_25/dropout/Mul
dropout_25/dropout/ShapeShape!max_pooling2d_17/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_25/dropout/ShapeÝ
/dropout_25/dropout/random_uniform/RandomUniformRandomUniform!dropout_25/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype021
/dropout_25/dropout/random_uniform/RandomUniform
!dropout_25/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB 2É?2#
!dropout_25/dropout/GreaterEqual/yò
dropout_25/dropout/GreaterEqualGreaterEqual8dropout_25/dropout/random_uniform/RandomUniform:output:0*dropout_25/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
dropout_25/dropout/GreaterEqual¨
dropout_25/dropout/CastCast#dropout_25/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_25/dropout/Cast®
dropout_25/dropout/Mul_1Muldropout_25/dropout/Mul:z:0dropout_25/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_25/dropout/Mul_1³
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_26/Conv2D/ReadVariableOpØ
conv2d_26/Conv2DConv2Ddropout_25/dropout/Mul_1:z:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_26/Conv2Dª
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp°
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_26/BiasAdd~
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_26/ReluÃ
5batch_normalization_26/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          27
5batch_normalization_26/moments/mean/reduction_indicesò
#batch_normalization_26/moments/meanMeanconv2d_26/Relu:activations:0>batch_normalization_26/moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(2%
#batch_normalization_26/moments/meanÉ
+batch_normalization_26/moments/StopGradientStopGradient,batch_normalization_26/moments/mean:output:0*
T0*&
_output_shapes
:@2-
+batch_normalization_26/moments/StopGradient
0batch_normalization_26/moments/SquaredDifferenceSquaredDifferenceconv2d_26/Relu:activations:04batch_normalization_26/moments/StopGradient:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@22
0batch_normalization_26/moments/SquaredDifferenceË
9batch_normalization_26/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2;
9batch_normalization_26/moments/variance/reduction_indices
'batch_normalization_26/moments/varianceMean4batch_normalization_26/moments/SquaredDifference:z:0Bbatch_normalization_26/moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(2)
'batch_normalization_26/moments/varianceÇ
&batch_normalization_26/moments/SqueezeSqueeze,batch_normalization_26/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2(
&batch_normalization_26/moments/SqueezeÏ
(batch_normalization_26/moments/Squeeze_1Squeeze0batch_normalization_26/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2*
(batch_normalization_26/moments/Squeeze_1æ
,batch_normalization_26/AssignMovingAvg/decayConst*C
_class9
75loc:@batch_normalization_26/AssignMovingAvg/534955377*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_26/AssignMovingAvg/decay
+batch_normalization_26/AssignMovingAvg/CastCast5batch_normalization_26/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*C
_class9
75loc:@batch_normalization_26/AssignMovingAvg/534955377*
_output_shapes
: 2-
+batch_normalization_26/AssignMovingAvg/CastÛ
5batch_normalization_26/AssignMovingAvg/ReadVariableOpReadVariableOp0batch_normalization_26_assignmovingavg_534955377*
_output_shapes
:@*
dtype027
5batch_normalization_26/AssignMovingAvg/ReadVariableOp¹
*batch_normalization_26/AssignMovingAvg/subSub=batch_normalization_26/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_26/moments/Squeeze:output:0*
T0*C
_class9
75loc:@batch_normalization_26/AssignMovingAvg/534955377*
_output_shapes
:@2,
*batch_normalization_26/AssignMovingAvg/subª
*batch_normalization_26/AssignMovingAvg/mulMul.batch_normalization_26/AssignMovingAvg/sub:z:0/batch_normalization_26/AssignMovingAvg/Cast:y:0*
T0*C
_class9
75loc:@batch_normalization_26/AssignMovingAvg/534955377*
_output_shapes
:@2,
*batch_normalization_26/AssignMovingAvg/mul
:batch_normalization_26/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp0batch_normalization_26_assignmovingavg_534955377.batch_normalization_26/AssignMovingAvg/mul:z:06^batch_normalization_26/AssignMovingAvg/ReadVariableOp*C
_class9
75loc:@batch_normalization_26/AssignMovingAvg/534955377*
_output_shapes
 *
dtype02<
:batch_normalization_26/AssignMovingAvg/AssignSubVariableOpì
.batch_normalization_26/AssignMovingAvg_1/decayConst*E
_class;
97loc:@batch_normalization_26/AssignMovingAvg_1/534955384*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_26/AssignMovingAvg_1/decay
-batch_normalization_26/AssignMovingAvg_1/CastCast7batch_normalization_26/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*E
_class;
97loc:@batch_normalization_26/AssignMovingAvg_1/534955384*
_output_shapes
: 2/
-batch_normalization_26/AssignMovingAvg_1/Castá
7batch_normalization_26/AssignMovingAvg_1/ReadVariableOpReadVariableOp2batch_normalization_26_assignmovingavg_1_534955384*
_output_shapes
:@*
dtype029
7batch_normalization_26/AssignMovingAvg_1/ReadVariableOpÃ
,batch_normalization_26/AssignMovingAvg_1/subSub?batch_normalization_26/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_26/moments/Squeeze_1:output:0*
T0*E
_class;
97loc:@batch_normalization_26/AssignMovingAvg_1/534955384*
_output_shapes
:@2.
,batch_normalization_26/AssignMovingAvg_1/sub´
,batch_normalization_26/AssignMovingAvg_1/mulMul0batch_normalization_26/AssignMovingAvg_1/sub:z:01batch_normalization_26/AssignMovingAvg_1/Cast:y:0*
T0*E
_class;
97loc:@batch_normalization_26/AssignMovingAvg_1/534955384*
_output_shapes
:@2.
,batch_normalization_26/AssignMovingAvg_1/mul
<batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp2batch_normalization_26_assignmovingavg_1_5349553840batch_normalization_26/AssignMovingAvg_1/mul:z:08^batch_normalization_26/AssignMovingAvg_1/ReadVariableOp*E
_class;
97loc:@batch_normalization_26/AssignMovingAvg_1/534955384*
_output_shapes
 *
dtype02>
<batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2(
&batch_normalization_26/batchnorm/add/yÞ
$batch_normalization_26/batchnorm/addAddV21batch_normalization_26/moments/Squeeze_1:output:0/batch_normalization_26/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2&
$batch_normalization_26/batchnorm/add¨
&batch_normalization_26/batchnorm/RsqrtRsqrt(batch_normalization_26/batchnorm/add:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_26/batchnorm/Rsqrtã
3batch_normalization_26/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_26_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3batch_normalization_26/batchnorm/mul/ReadVariableOpá
$batch_normalization_26/batchnorm/mulMul*batch_normalization_26/batchnorm/Rsqrt:y:0;batch_normalization_26/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2&
$batch_normalization_26/batchnorm/mulÙ
&batch_normalization_26/batchnorm/mul_1Mulconv2d_26/Relu:activations:0(batch_normalization_26/batchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&batch_normalization_26/batchnorm/mul_1×
&batch_normalization_26/batchnorm/mul_2Mul/batch_normalization_26/moments/Squeeze:output:0(batch_normalization_26/batchnorm/mul:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_26/batchnorm/mul_2×
/batch_normalization_26/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_26_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/batch_normalization_26/batchnorm/ReadVariableOpÝ
$batch_normalization_26/batchnorm/subSub7batch_normalization_26/batchnorm/ReadVariableOp:value:0*batch_normalization_26/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2&
$batch_normalization_26/batchnorm/subé
&batch_normalization_26/batchnorm/add_1AddV2*batch_normalization_26/batchnorm/mul_1:z:0(batch_normalization_26/batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&batch_normalization_26/batchnorm/add_1}
dropout_26/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB 2·mÛ¶mÛö?2
dropout_26/dropout/ConstÀ
dropout_26/dropout/MulMul*batch_normalization_26/batchnorm/add_1:z:0!dropout_26/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_26/dropout/Mul
dropout_26/dropout/ShapeShape*batch_normalization_26/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dropout_26/dropout/ShapeÝ
/dropout_26/dropout/random_uniform/RandomUniformRandomUniform!dropout_26/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype021
/dropout_26/dropout/random_uniform/RandomUniform
!dropout_26/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB 2433333Ó?2#
!dropout_26/dropout/GreaterEqual/yò
dropout_26/dropout/GreaterEqualGreaterEqual8dropout_26/dropout/random_uniform/RandomUniform:output:0*dropout_26/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
dropout_26/dropout/GreaterEqual¨
dropout_26/dropout/CastCast#dropout_26/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_26/dropout/Cast®
dropout_26/dropout/Mul_1Muldropout_26/dropout/Mul:z:0dropout_26/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_26/dropout/Mul_1s
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
flatten_8/Const
flatten_8/ReshapeReshapedropout_26/dropout/Mul_1:z:0flatten_8/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
flatten_8/Reshape¦
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	À
*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMulflatten_8/Reshape:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_8/MatMul¤
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_8/BiasAdd/ReadVariableOp¡
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_8/BiasAddà
IdentityIdentitydense_8/BiasAdd:output:0;^batch_normalization_24/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_25/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_26/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::2x
:batch_normalization_24/AssignMovingAvg/AssignSubVariableOp:batch_normalization_24/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_25/AssignMovingAvg/AssignSubVariableOp:batch_normalization_25/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_26/AssignMovingAvg/AssignSubVariableOp:batch_normalization_26/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOp:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  

_user_specified_namex
³

U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_534956104

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul~
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
-
Ô
U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_534955654

inputs
assignmovingavg_534955627
assignmovingavg_1_534955634)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
: 2
moments/StopGradient¬
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"moments/variance/reduction_indicesº
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1¡
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/534955627*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay²
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*,
_class"
 loc:@AssignMovingAvg/534955627*
_output_shapes
: 2
AssignMovingAvg/Cast
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_534955627*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpÆ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/534955627*
_output_shapes
: 2
AssignMovingAvg/sub·
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*,
_class"
 loc:@AssignMovingAvg/534955627*
_output_shapes
: 2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_534955627AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/534955627*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/534955634*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decayº
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*.
_class$
" loc:@AssignMovingAvg_1/534955634*
_output_shapes
: 2
AssignMovingAvg_1/Cast
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_534955634*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpÐ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534955634*
_output_shapes
: 2
AssignMovingAvg_1/subÁ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534955634*
_output_shapes
: 2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_534955634AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/534955634*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul~
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1½
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Á
J
.__inference_dropout_26_layer_call_fn_534956241

inputs
identityÔ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_26_layer_call_and_return_conditional_losses_5349545592
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ñ

.__inference_conv_net_9_layer_call_fn_534955259
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_conv_net_9_layer_call_and_return_conditional_losses_5349548322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_1
À
d
H__inference_flatten_8_layer_call_and_return_conditional_losses_534954578

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
l
Ó	
I__inference_conv_net_9_layer_call_and_return_conditional_losses_534955506
x,
(conv2d_24_conv2d_readvariableop_resource-
)conv2d_24_biasadd_readvariableop_resource<
8batch_normalization_24_batchnorm_readvariableop_resource@
<batch_normalization_24_batchnorm_mul_readvariableop_resource>
:batch_normalization_24_batchnorm_readvariableop_1_resource>
:batch_normalization_24_batchnorm_readvariableop_2_resource,
(conv2d_25_conv2d_readvariableop_resource-
)conv2d_25_biasadd_readvariableop_resource<
8batch_normalization_25_batchnorm_readvariableop_resource@
<batch_normalization_25_batchnorm_mul_readvariableop_resource>
:batch_normalization_25_batchnorm_readvariableop_1_resource>
:batch_normalization_25_batchnorm_readvariableop_2_resource,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource<
8batch_normalization_26_batchnorm_readvariableop_resource@
<batch_normalization_26_batchnorm_mul_readvariableop_resource>
:batch_normalization_26_batchnorm_readvariableop_1_resource>
:batch_normalization_26_batchnorm_readvariableop_2_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource
identity³
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_24/Conv2D/ReadVariableOp½
conv2d_24/Conv2DConv2Dx'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_24/Conv2Dª
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp°
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_24/BiasAdd~
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_24/Relu×
/batch_normalization_24/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_24_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_24/batchnorm/ReadVariableOp
&batch_normalization_24/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2(
&batch_normalization_24/batchnorm/add/yä
$batch_normalization_24/batchnorm/addAddV27batch_normalization_24/batchnorm/ReadVariableOp:value:0/batch_normalization_24/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_24/batchnorm/add¨
&batch_normalization_24/batchnorm/RsqrtRsqrt(batch_normalization_24/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_24/batchnorm/Rsqrtã
3batch_normalization_24/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_24_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_24/batchnorm/mul/ReadVariableOpá
$batch_normalization_24/batchnorm/mulMul*batch_normalization_24/batchnorm/Rsqrt:y:0;batch_normalization_24/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_24/batchnorm/mulÙ
&batch_normalization_24/batchnorm/mul_1Mulconv2d_24/Relu:activations:0(batch_normalization_24/batchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&batch_normalization_24/batchnorm/mul_1Ý
1batch_normalization_24/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_24_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype023
1batch_normalization_24/batchnorm/ReadVariableOp_1á
&batch_normalization_24/batchnorm/mul_2Mul9batch_normalization_24/batchnorm/ReadVariableOp_1:value:0(batch_normalization_24/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_24/batchnorm/mul_2Ý
1batch_normalization_24/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_24_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype023
1batch_normalization_24/batchnorm/ReadVariableOp_2ß
$batch_normalization_24/batchnorm/subSub9batch_normalization_24/batchnorm/ReadVariableOp_2:value:0*batch_normalization_24/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_24/batchnorm/subé
&batch_normalization_24/batchnorm/add_1AddV2*batch_normalization_24/batchnorm/mul_1:z:0(batch_normalization_24/batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&batch_normalization_24/batchnorm/add_1á
max_pooling2d_16/MaxPoolMaxPool*batch_normalization_24/batchnorm/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_16/MaxPool
dropout_24/IdentityIdentity!max_pooling2d_16/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_24/Identity³
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_25/Conv2D/ReadVariableOpØ
conv2d_25/Conv2DConv2Ddropout_24/Identity:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *
paddingVALID*
strides
2
conv2d_25/Conv2Dª
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_25/BiasAdd/ReadVariableOp°
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
conv2d_25/BiasAdd~
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
conv2d_25/Relu×
/batch_normalization_25/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_25_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_25/batchnorm/ReadVariableOp
&batch_normalization_25/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2(
&batch_normalization_25/batchnorm/add/yä
$batch_normalization_25/batchnorm/addAddV27batch_normalization_25/batchnorm/ReadVariableOp:value:0/batch_normalization_25/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_25/batchnorm/add¨
&batch_normalization_25/batchnorm/RsqrtRsqrt(batch_normalization_25/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_25/batchnorm/Rsqrtã
3batch_normalization_25/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_25_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_25/batchnorm/mul/ReadVariableOpá
$batch_normalization_25/batchnorm/mulMul*batch_normalization_25/batchnorm/Rsqrt:y:0;batch_normalization_25/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_25/batchnorm/mulÙ
&batch_normalization_25/batchnorm/mul_1Mulconv2d_25/Relu:activations:0(batch_normalization_25/batchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2(
&batch_normalization_25/batchnorm/mul_1Ý
1batch_normalization_25/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_25_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype023
1batch_normalization_25/batchnorm/ReadVariableOp_1á
&batch_normalization_25/batchnorm/mul_2Mul9batch_normalization_25/batchnorm/ReadVariableOp_1:value:0(batch_normalization_25/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_25/batchnorm/mul_2Ý
1batch_normalization_25/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_25_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype023
1batch_normalization_25/batchnorm/ReadVariableOp_2ß
$batch_normalization_25/batchnorm/subSub9batch_normalization_25/batchnorm/ReadVariableOp_2:value:0*batch_normalization_25/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_25/batchnorm/subé
&batch_normalization_25/batchnorm/add_1AddV2*batch_normalization_25/batchnorm/mul_1:z:0(batch_normalization_25/batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2(
&batch_normalization_25/batchnorm/add_1á
max_pooling2d_17/MaxPoolMaxPool*batch_normalization_25/batchnorm/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_17/MaxPool
dropout_25/IdentityIdentity!max_pooling2d_17/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_25/Identity³
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_26/Conv2D/ReadVariableOpØ
conv2d_26/Conv2DConv2Ddropout_25/Identity:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_26/Conv2Dª
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp°
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_26/BiasAdd~
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_26/Relu×
/batch_normalization_26/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_26_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/batch_normalization_26/batchnorm/ReadVariableOp
&batch_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2(
&batch_normalization_26/batchnorm/add/yä
$batch_normalization_26/batchnorm/addAddV27batch_normalization_26/batchnorm/ReadVariableOp:value:0/batch_normalization_26/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2&
$batch_normalization_26/batchnorm/add¨
&batch_normalization_26/batchnorm/RsqrtRsqrt(batch_normalization_26/batchnorm/add:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_26/batchnorm/Rsqrtã
3batch_normalization_26/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_26_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3batch_normalization_26/batchnorm/mul/ReadVariableOpá
$batch_normalization_26/batchnorm/mulMul*batch_normalization_26/batchnorm/Rsqrt:y:0;batch_normalization_26/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2&
$batch_normalization_26/batchnorm/mulÙ
&batch_normalization_26/batchnorm/mul_1Mulconv2d_26/Relu:activations:0(batch_normalization_26/batchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&batch_normalization_26/batchnorm/mul_1Ý
1batch_normalization_26/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_26_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype023
1batch_normalization_26/batchnorm/ReadVariableOp_1á
&batch_normalization_26/batchnorm/mul_2Mul9batch_normalization_26/batchnorm/ReadVariableOp_1:value:0(batch_normalization_26/batchnorm/mul:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_26/batchnorm/mul_2Ý
1batch_normalization_26/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_26_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype023
1batch_normalization_26/batchnorm/ReadVariableOp_2ß
$batch_normalization_26/batchnorm/subSub9batch_normalization_26/batchnorm/ReadVariableOp_2:value:0*batch_normalization_26/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2&
$batch_normalization_26/batchnorm/subé
&batch_normalization_26/batchnorm/add_1AddV2*batch_normalization_26/batchnorm/mul_1:z:0(batch_normalization_26/batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&batch_normalization_26/batchnorm/add_1
dropout_26/IdentityIdentity*batch_normalization_26/batchnorm/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_26/Identitys
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
flatten_8/Const
flatten_8/ReshapeReshapedropout_26/Identity:output:0flatten_8/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
flatten_8/Reshape¦
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	À
*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMulflatten_8/Reshape:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_8/MatMul¤
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_8/BiasAdd/ReadVariableOp¡
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_8/BiasAddl
IdentityIdentitydense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ  :::::::::::::::::::::R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  

_user_specified_namex


U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_534956188

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Í
g
.__inference_dropout_25_layer_call_fn_534956021

inputs
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_25_layer_call_and_return_conditional_losses_5349544042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
.
Ô
U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_534956168

inputs
assignmovingavg_534956141
assignmovingavg_1_534956148)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:@2
moments/StopGradient¾
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"moments/variance/reduction_indicesº
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1¡
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/534956141*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay²
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*,
_class"
 loc:@AssignMovingAvg/534956141*
_output_shapes
: 2
AssignMovingAvg/Cast
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_534956141*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpÆ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/534956141*
_output_shapes
:@2
AssignMovingAvg/sub·
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*,
_class"
 loc:@AssignMovingAvg/534956141*
_output_shapes
:@2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_534956141AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/534956141*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/534956148*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decayº
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*.
_class$
" loc:@AssignMovingAvg_1/534956148*
_output_shapes
: 2
AssignMovingAvg_1/Cast
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_534956148*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÐ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534956148*
_output_shapes
:@2
AssignMovingAvg_1/subÁ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534956148*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_534956148AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/534956148*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
batchnorm/add_1Ï
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ì
g
I__inference_dropout_24_layer_call_and_return_conditional_losses_534955801

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¹

.__inference_conv_net_9_layer_call_fn_534955551
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*0
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_conv_net_9_layer_call_and_return_conditional_losses_5349547302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  

_user_specified_namex
³

U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_534954204

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul~
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ :::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ä
­
:__inference_batch_normalization_26_layer_call_fn_534956130

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *^
fYRW
U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_5349545062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
â
­
:__inference_batch_normalization_24_layer_call_fn_534955687

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *^
fYRW
U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_5349541842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

k
O__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_534953966

inputs
identity¶
MaxPoolMaxPoolinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
.
Ô
U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_534955738

inputs
assignmovingavg_534955711
assignmovingavg_1_534955718)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
: 2
moments/StopGradient¾
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"moments/variance/reduction_indicesº
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1¡
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/534955711*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay²
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*,
_class"
 loc:@AssignMovingAvg/534955711*
_output_shapes
: 2
AssignMovingAvg/Cast
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_534955711*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpÆ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/534955711*
_output_shapes
: 2
AssignMovingAvg/sub·
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*,
_class"
 loc:@AssignMovingAvg/534955711*
_output_shapes
: 2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_534955711AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/534955711*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/534955718*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decayº
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*.
_class$
" loc:@AssignMovingAvg_1/534955718*
_output_shapes
: 2
AssignMovingAvg_1/Cast
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_534955718*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpÐ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534955718*
_output_shapes
: 2
AssignMovingAvg_1/subÁ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534955718*
_output_shapes
: 2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_534955718AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/534955718*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1Ï
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ª
­
:__inference_batch_normalization_26_layer_call_fn_534956201

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *^
fYRW
U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_5349540722
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
°
H__inference_conv2d_24_layer_call_and_return_conditional_losses_534955607

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ  :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ò
®
F__inference_dense_8_layer_call_and_return_conditional_losses_534956262

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	À
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
º
P
4__inference_max_pooling2d_16_layer_call_fn_534953816

inputs
identityõ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_5349538102
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
I
-__inference_flatten_8_layer_call_fn_534956252

inputs
identityÌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_flatten_8_layer_call_and_return_conditional_losses_5349545782
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Í
g
.__inference_dropout_24_layer_call_fn_534955806

inputs
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_24_layer_call_and_return_conditional_losses_5349542532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
-
Ô
U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_534956084

inputs
assignmovingavg_534956057
assignmovingavg_1_534956064)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:@2
moments/StopGradient¬
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"moments/variance/reduction_indicesº
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1¡
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/534956057*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay²
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*,
_class"
 loc:@AssignMovingAvg/534956057*
_output_shapes
: 2
AssignMovingAvg/Cast
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_534956057*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpÆ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/534956057*
_output_shapes
:@2
AssignMovingAvg/sub·
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*,
_class"
 loc:@AssignMovingAvg/534956057*
_output_shapes
:@2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_534956057AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/534956057*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/534956064*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decayº
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*.
_class$
" loc:@AssignMovingAvg_1/534956064*
_output_shapes
: 2
AssignMovingAvg_1/Cast
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_534956064*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÐ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534956064*
_output_shapes
:@2
AssignMovingAvg_1/subÁ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534956064*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_534956064AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/534956064*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul~
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
batchnorm/add_1½
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_534953949

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
-
Ô
U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_534954335

inputs
assignmovingavg_534954308
assignmovingavg_1_534954315)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
: 2
moments/StopGradient¬
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"moments/variance/reduction_indicesº
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1¡
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/534954308*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay²
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*,
_class"
 loc:@AssignMovingAvg/534954308*
_output_shapes
: 2
AssignMovingAvg/Cast
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_534954308*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpÆ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/534954308*
_output_shapes
: 2
AssignMovingAvg/sub·
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*,
_class"
 loc:@AssignMovingAvg/534954308*
_output_shapes
: 2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_534954308AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/534954308*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/534954315*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decayº
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*.
_class$
" loc:@AssignMovingAvg_1/534954315*
_output_shapes
: 2
AssignMovingAvg_1/Cast
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_534954315*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpÐ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534954315*
_output_shapes
: 2
AssignMovingAvg_1/subÁ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534954315*
_output_shapes
: 2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_534954315AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/534954315*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul~
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
batchnorm/add_1½
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ

 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
 
_user_specified_nameinputs


U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_534955889

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
â
­
:__inference_batch_normalization_25_layer_call_fn_534955986

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *^
fYRW
U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5349543352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ

 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
 
_user_specified_nameinputs
ª
­
:__inference_batch_normalization_24_layer_call_fn_534955771

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *^
fYRW
U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_5349537602
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ì
g
I__inference_dropout_26_layer_call_and_return_conditional_losses_534954559

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ä
­
:__inference_batch_normalization_25_layer_call_fn_534955999

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *^
fYRW
U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5349543552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ

 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
 
_user_specified_nameinputs
¬
­
:__inference_batch_normalization_25_layer_call_fn_534955915

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *^
fYRW
U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5349539492
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
.
Ô
U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_534953916

inputs
assignmovingavg_534953889
assignmovingavg_1_534953896)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
: 2
moments/StopGradient¾
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"moments/variance/reduction_indicesº
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1¡
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/534953889*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay²
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*,
_class"
 loc:@AssignMovingAvg/534953889*
_output_shapes
: 2
AssignMovingAvg/Cast
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_534953889*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpÆ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/534953889*
_output_shapes
: 2
AssignMovingAvg/sub·
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*,
_class"
 loc:@AssignMovingAvg/534953889*
_output_shapes
: 2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_534953889AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/534953889*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/534953896*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decayº
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*.
_class$
" loc:@AssignMovingAvg_1/534953896*
_output_shapes
: 2
AssignMovingAvg_1/Cast
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_534953896*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpÐ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534953896*
_output_shapes
: 2
AssignMovingAvg_1/subÁ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534953896*
_output_shapes
: 2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_534953896AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/534953896*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1Ï
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ï
h
I__inference_dropout_25_layer_call_and_return_conditional_losses_534956011

inputs
identityg
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB 2      ô?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02&
$dropout/random_uniform/RandomUniformy
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB 2É?2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ì
g
I__inference_dropout_26_layer_call_and_return_conditional_losses_534956231

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ï
h
I__inference_dropout_24_layer_call_and_return_conditional_losses_534955796

inputs
identityg
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB 2rÇqÇñ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02&
$dropout/random_uniform/RandomUniformy
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB 2¹?2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
³

U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_534954355

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul~
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ

 :::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
 
_user_specified_nameinputs
¶6
Ð
"__inference__traced_save_534956354
file_prefix:
6savev2_conv_net_9_conv2d_24_kernel_read_readvariableop8
4savev2_conv_net_9_conv2d_24_bias_read_readvariableopF
Bsavev2_conv_net_9_batch_normalization_24_gamma_read_readvariableopE
Asavev2_conv_net_9_batch_normalization_24_beta_read_readvariableopL
Hsavev2_conv_net_9_batch_normalization_24_moving_mean_read_readvariableopP
Lsavev2_conv_net_9_batch_normalization_24_moving_variance_read_readvariableop:
6savev2_conv_net_9_conv2d_25_kernel_read_readvariableop8
4savev2_conv_net_9_conv2d_25_bias_read_readvariableopF
Bsavev2_conv_net_9_batch_normalization_25_gamma_read_readvariableopE
Asavev2_conv_net_9_batch_normalization_25_beta_read_readvariableopL
Hsavev2_conv_net_9_batch_normalization_25_moving_mean_read_readvariableopP
Lsavev2_conv_net_9_batch_normalization_25_moving_variance_read_readvariableop:
6savev2_conv_net_9_conv2d_26_kernel_read_readvariableop8
4savev2_conv_net_9_conv2d_26_bias_read_readvariableopF
Bsavev2_conv_net_9_batch_normalization_26_gamma_read_readvariableopE
Asavev2_conv_net_9_batch_normalization_26_beta_read_readvariableopL
Hsavev2_conv_net_9_batch_normalization_26_moving_mean_read_readvariableopP
Lsavev2_conv_net_9_batch_normalization_26_moving_variance_read_readvariableop8
4savev2_conv_net_9_dense_8_kernel_read_readvariableop6
2savev2_conv_net_9_dense_8_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_968a7078479240cbb14c373fb315eab7/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename¶
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*È
value¾B»B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB,batch_norm1/gamma/.ATTRIBUTES/VARIABLE_VALUEB+batch_norm1/beta/.ATTRIBUTES/VARIABLE_VALUEB2batch_norm1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB,batch_norm2/gamma/.ATTRIBUTES/VARIABLE_VALUEB+batch_norm2/beta/.ATTRIBUTES/VARIABLE_VALUEB2batch_norm2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB,batch_norm3/gamma/.ATTRIBUTES/VARIABLE_VALUEB+batch_norm3/beta/.ATTRIBUTES/VARIABLE_VALUEB2batch_norm3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names²
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesß
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_conv_net_9_conv2d_24_kernel_read_readvariableop4savev2_conv_net_9_conv2d_24_bias_read_readvariableopBsavev2_conv_net_9_batch_normalization_24_gamma_read_readvariableopAsavev2_conv_net_9_batch_normalization_24_beta_read_readvariableopHsavev2_conv_net_9_batch_normalization_24_moving_mean_read_readvariableopLsavev2_conv_net_9_batch_normalization_24_moving_variance_read_readvariableop6savev2_conv_net_9_conv2d_25_kernel_read_readvariableop4savev2_conv_net_9_conv2d_25_bias_read_readvariableopBsavev2_conv_net_9_batch_normalization_25_gamma_read_readvariableopAsavev2_conv_net_9_batch_normalization_25_beta_read_readvariableopHsavev2_conv_net_9_batch_normalization_25_moving_mean_read_readvariableopLsavev2_conv_net_9_batch_normalization_25_moving_variance_read_readvariableop6savev2_conv_net_9_conv2d_26_kernel_read_readvariableop4savev2_conv_net_9_conv2d_26_bias_read_readvariableopBsavev2_conv_net_9_batch_normalization_26_gamma_read_readvariableopAsavev2_conv_net_9_batch_normalization_26_beta_read_readvariableopHsavev2_conv_net_9_batch_normalization_26_moving_mean_read_readvariableopLsavev2_conv_net_9_batch_normalization_26_moving_variance_read_readvariableop4savev2_conv_net_9_dense_8_kernel_read_readvariableop2savev2_conv_net_9_dense_8_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *#
dtypes
22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*º
_input_shapes¨
¥: : : : : : : :  : : : : : : @:@:@:@:@:@:	À
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	À
: 

_output_shapes
:
:

_output_shapes
: 
â
­
:__inference_batch_normalization_26_layer_call_fn_534956117

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *^
fYRW
U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_5349544862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ä
­
:__inference_batch_normalization_24_layer_call_fn_534955700

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *^
fYRW
U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_5349542042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


-__inference_conv2d_26_layer_call_fn_534956046

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_conv2d_26_layer_call_and_return_conditional_losses_5349544332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_534953793

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ªA

I__inference_conv_net_9_layer_call_and_return_conditional_losses_534954832
x
conv2d_24_534954778
conv2d_24_534954780$
 batch_normalization_24_534954783$
 batch_normalization_24_534954785$
 batch_normalization_24_534954787$
 batch_normalization_24_534954789
conv2d_25_534954794
conv2d_25_534954796$
 batch_normalization_25_534954799$
 batch_normalization_25_534954801$
 batch_normalization_25_534954803$
 batch_normalization_25_534954805
conv2d_26_534954810
conv2d_26_534954812$
 batch_normalization_26_534954815$
 batch_normalization_26_534954817$
 batch_normalization_26_534954819$
 batch_normalization_26_534954821
dense_8_534954826
dense_8_534954828
identity¢.batch_normalization_24/StatefulPartitionedCall¢.batch_normalization_25/StatefulPartitionedCall¢.batch_normalization_26/StatefulPartitionedCall¢!conv2d_24/StatefulPartitionedCall¢!conv2d_25/StatefulPartitionedCall¢!conv2d_26/StatefulPartitionedCall¢dense_8/StatefulPartitionedCallª
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCallxconv2d_24_534954778conv2d_24_534954780*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_conv2d_24_layer_call_and_return_conditional_losses_5349541312#
!conv2d_24/StatefulPartitionedCallÜ
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0 batch_normalization_24_534954783 batch_normalization_24_534954785 batch_normalization_24_534954787 batch_normalization_24_534954789*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *^
fYRW
U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_53495420420
.batch_normalization_24/StatefulPartitionedCall­
 max_pooling2d_16/PartitionedCallPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_5349538102"
 max_pooling2d_16/PartitionedCall
dropout_24/PartitionedCallPartitionedCall)max_pooling2d_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_24_layer_call_and_return_conditional_losses_5349542582
dropout_24/PartitionedCallÌ
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall#dropout_24/PartitionedCall:output:0conv2d_25_534954794conv2d_25_534954796*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_conv2d_25_layer_call_and_return_conditional_losses_5349542822#
!conv2d_25/StatefulPartitionedCallÜ
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0 batch_normalization_25_534954799 batch_normalization_25_534954801 batch_normalization_25_534954803 batch_normalization_25_534954805*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *^
fYRW
U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_53495435520
.batch_normalization_25/StatefulPartitionedCall­
 max_pooling2d_17/PartitionedCallPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_5349539662"
 max_pooling2d_17/PartitionedCall
dropout_25/PartitionedCallPartitionedCall)max_pooling2d_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_25_layer_call_and_return_conditional_losses_5349544092
dropout_25/PartitionedCallÌ
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_25/PartitionedCall:output:0conv2d_26_534954810conv2d_26_534954812*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_conv2d_26_layer_call_and_return_conditional_losses_5349544332#
!conv2d_26/StatefulPartitionedCallÜ
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0 batch_normalization_26_534954815 batch_normalization_26_534954817 batch_normalization_26_534954819 batch_normalization_26_534954821*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *^
fYRW
U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_53495450620
.batch_normalization_26/StatefulPartitionedCall
dropout_26/PartitionedCallPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_26_layer_call_and_return_conditional_losses_5349545592
dropout_26/PartitionedCallý
flatten_8/PartitionedCallPartitionedCall#dropout_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_flatten_8_layer_call_and_return_conditional_losses_5349545782
flatten_8/PartitionedCall¹
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_8_534954826dense_8_534954828*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_8_layer_call_and_return_conditional_losses_5349545962!
dense_8/StatefulPartitionedCall
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_26/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  

_user_specified_namex
.
Ô
U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_534954072

inputs
assignmovingavg_534954045
assignmovingavg_1_534954052)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:@2
moments/StopGradient¾
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"moments/variance/reduction_indicesº
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1¡
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/534954045*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay²
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*,
_class"
 loc:@AssignMovingAvg/534954045*
_output_shapes
: 2
AssignMovingAvg/Cast
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_534954045*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpÆ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/534954045*
_output_shapes
:@2
AssignMovingAvg/sub·
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*,
_class"
 loc:@AssignMovingAvg/534954045*
_output_shapes
:@2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_534954045AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/534954045*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/534954052*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decayº
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*.
_class$
" loc:@AssignMovingAvg_1/534954052*
_output_shapes
: 2
AssignMovingAvg_1/Cast
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_534954052*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÐ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534954052*
_output_shapes
:@2
AssignMovingAvg_1/subÁ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/534954052*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_534954052AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/534954052*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
batchnorm/add_1Ï
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ì

I__inference_conv_net_9_layer_call_and_return_conditional_losses_534955083
input_1,
(conv2d_24_conv2d_readvariableop_resource-
)conv2d_24_biasadd_readvariableop_resource4
0batch_normalization_24_assignmovingavg_5349549406
2batch_normalization_24_assignmovingavg_1_534954947@
<batch_normalization_24_batchnorm_mul_readvariableop_resource<
8batch_normalization_24_batchnorm_readvariableop_resource,
(conv2d_25_conv2d_readvariableop_resource-
)conv2d_25_biasadd_readvariableop_resource4
0batch_normalization_25_assignmovingavg_5349549906
2batch_normalization_25_assignmovingavg_1_534954997@
<batch_normalization_25_batchnorm_mul_readvariableop_resource<
8batch_normalization_25_batchnorm_readvariableop_resource,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource4
0batch_normalization_26_assignmovingavg_5349550406
2batch_normalization_26_assignmovingavg_1_534955047@
<batch_normalization_26_batchnorm_mul_readvariableop_resource<
8batch_normalization_26_batchnorm_readvariableop_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource
identity¢:batch_normalization_24/AssignMovingAvg/AssignSubVariableOp¢<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp¢:batch_normalization_25/AssignMovingAvg/AssignSubVariableOp¢<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp¢:batch_normalization_26/AssignMovingAvg/AssignSubVariableOp¢<batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOp³
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_24/Conv2D/ReadVariableOpÃ
conv2d_24/Conv2DConv2Dinput_1'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_24/Conv2Dª
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp°
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_24/BiasAdd~
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_24/ReluÃ
5batch_normalization_24/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          27
5batch_normalization_24/moments/mean/reduction_indicesò
#batch_normalization_24/moments/meanMeanconv2d_24/Relu:activations:0>batch_normalization_24/moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2%
#batch_normalization_24/moments/meanÉ
+batch_normalization_24/moments/StopGradientStopGradient,batch_normalization_24/moments/mean:output:0*
T0*&
_output_shapes
: 2-
+batch_normalization_24/moments/StopGradient
0batch_normalization_24/moments/SquaredDifferenceSquaredDifferenceconv2d_24/Relu:activations:04batch_normalization_24/moments/StopGradient:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0batch_normalization_24/moments/SquaredDifferenceË
9batch_normalization_24/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2;
9batch_normalization_24/moments/variance/reduction_indices
'batch_normalization_24/moments/varianceMean4batch_normalization_24/moments/SquaredDifference:z:0Bbatch_normalization_24/moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2)
'batch_normalization_24/moments/varianceÇ
&batch_normalization_24/moments/SqueezeSqueeze,batch_normalization_24/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2(
&batch_normalization_24/moments/SqueezeÏ
(batch_normalization_24/moments/Squeeze_1Squeeze0batch_normalization_24/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2*
(batch_normalization_24/moments/Squeeze_1æ
,batch_normalization_24/AssignMovingAvg/decayConst*C
_class9
75loc:@batch_normalization_24/AssignMovingAvg/534954940*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_24/AssignMovingAvg/decay
+batch_normalization_24/AssignMovingAvg/CastCast5batch_normalization_24/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*C
_class9
75loc:@batch_normalization_24/AssignMovingAvg/534954940*
_output_shapes
: 2-
+batch_normalization_24/AssignMovingAvg/CastÛ
5batch_normalization_24/AssignMovingAvg/ReadVariableOpReadVariableOp0batch_normalization_24_assignmovingavg_534954940*
_output_shapes
: *
dtype027
5batch_normalization_24/AssignMovingAvg/ReadVariableOp¹
*batch_normalization_24/AssignMovingAvg/subSub=batch_normalization_24/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_24/moments/Squeeze:output:0*
T0*C
_class9
75loc:@batch_normalization_24/AssignMovingAvg/534954940*
_output_shapes
: 2,
*batch_normalization_24/AssignMovingAvg/subª
*batch_normalization_24/AssignMovingAvg/mulMul.batch_normalization_24/AssignMovingAvg/sub:z:0/batch_normalization_24/AssignMovingAvg/Cast:y:0*
T0*C
_class9
75loc:@batch_normalization_24/AssignMovingAvg/534954940*
_output_shapes
: 2,
*batch_normalization_24/AssignMovingAvg/mul
:batch_normalization_24/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp0batch_normalization_24_assignmovingavg_534954940.batch_normalization_24/AssignMovingAvg/mul:z:06^batch_normalization_24/AssignMovingAvg/ReadVariableOp*C
_class9
75loc:@batch_normalization_24/AssignMovingAvg/534954940*
_output_shapes
 *
dtype02<
:batch_normalization_24/AssignMovingAvg/AssignSubVariableOpì
.batch_normalization_24/AssignMovingAvg_1/decayConst*E
_class;
97loc:@batch_normalization_24/AssignMovingAvg_1/534954947*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_24/AssignMovingAvg_1/decay
-batch_normalization_24/AssignMovingAvg_1/CastCast7batch_normalization_24/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*E
_class;
97loc:@batch_normalization_24/AssignMovingAvg_1/534954947*
_output_shapes
: 2/
-batch_normalization_24/AssignMovingAvg_1/Castá
7batch_normalization_24/AssignMovingAvg_1/ReadVariableOpReadVariableOp2batch_normalization_24_assignmovingavg_1_534954947*
_output_shapes
: *
dtype029
7batch_normalization_24/AssignMovingAvg_1/ReadVariableOpÃ
,batch_normalization_24/AssignMovingAvg_1/subSub?batch_normalization_24/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_24/moments/Squeeze_1:output:0*
T0*E
_class;
97loc:@batch_normalization_24/AssignMovingAvg_1/534954947*
_output_shapes
: 2.
,batch_normalization_24/AssignMovingAvg_1/sub´
,batch_normalization_24/AssignMovingAvg_1/mulMul0batch_normalization_24/AssignMovingAvg_1/sub:z:01batch_normalization_24/AssignMovingAvg_1/Cast:y:0*
T0*E
_class;
97loc:@batch_normalization_24/AssignMovingAvg_1/534954947*
_output_shapes
: 2.
,batch_normalization_24/AssignMovingAvg_1/mul
<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp2batch_normalization_24_assignmovingavg_1_5349549470batch_normalization_24/AssignMovingAvg_1/mul:z:08^batch_normalization_24/AssignMovingAvg_1/ReadVariableOp*E
_class;
97loc:@batch_normalization_24/AssignMovingAvg_1/534954947*
_output_shapes
 *
dtype02>
<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_24/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2(
&batch_normalization_24/batchnorm/add/yÞ
$batch_normalization_24/batchnorm/addAddV21batch_normalization_24/moments/Squeeze_1:output:0/batch_normalization_24/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_24/batchnorm/add¨
&batch_normalization_24/batchnorm/RsqrtRsqrt(batch_normalization_24/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_24/batchnorm/Rsqrtã
3batch_normalization_24/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_24_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_24/batchnorm/mul/ReadVariableOpá
$batch_normalization_24/batchnorm/mulMul*batch_normalization_24/batchnorm/Rsqrt:y:0;batch_normalization_24/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_24/batchnorm/mulÙ
&batch_normalization_24/batchnorm/mul_1Mulconv2d_24/Relu:activations:0(batch_normalization_24/batchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&batch_normalization_24/batchnorm/mul_1×
&batch_normalization_24/batchnorm/mul_2Mul/batch_normalization_24/moments/Squeeze:output:0(batch_normalization_24/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_24/batchnorm/mul_2×
/batch_normalization_24/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_24_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_24/batchnorm/ReadVariableOpÝ
$batch_normalization_24/batchnorm/subSub7batch_normalization_24/batchnorm/ReadVariableOp:value:0*batch_normalization_24/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_24/batchnorm/subé
&batch_normalization_24/batchnorm/add_1AddV2*batch_normalization_24/batchnorm/mul_1:z:0(batch_normalization_24/batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&batch_normalization_24/batchnorm/add_1á
max_pooling2d_16/MaxPoolMaxPool*batch_normalization_24/batchnorm/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_16/MaxPool}
dropout_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB 2rÇqÇñ?2
dropout_24/dropout/Const·
dropout_24/dropout/MulMul!max_pooling2d_16/MaxPool:output:0!dropout_24/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_24/dropout/Mul
dropout_24/dropout/ShapeShape!max_pooling2d_16/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_24/dropout/ShapeÝ
/dropout_24/dropout/random_uniform/RandomUniformRandomUniform!dropout_24/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype021
/dropout_24/dropout/random_uniform/RandomUniform
!dropout_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB 2¹?2#
!dropout_24/dropout/GreaterEqual/yò
dropout_24/dropout/GreaterEqualGreaterEqual8dropout_24/dropout/random_uniform/RandomUniform:output:0*dropout_24/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
dropout_24/dropout/GreaterEqual¨
dropout_24/dropout/CastCast#dropout_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_24/dropout/Cast®
dropout_24/dropout/Mul_1Muldropout_24/dropout/Mul:z:0dropout_24/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_24/dropout/Mul_1³
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_25/Conv2D/ReadVariableOpØ
conv2d_25/Conv2DConv2Ddropout_24/dropout/Mul_1:z:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *
paddingVALID*
strides
2
conv2d_25/Conv2Dª
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_25/BiasAdd/ReadVariableOp°
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
conv2d_25/BiasAdd~
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
conv2d_25/ReluÃ
5batch_normalization_25/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          27
5batch_normalization_25/moments/mean/reduction_indicesò
#batch_normalization_25/moments/meanMeanconv2d_25/Relu:activations:0>batch_normalization_25/moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2%
#batch_normalization_25/moments/meanÉ
+batch_normalization_25/moments/StopGradientStopGradient,batch_normalization_25/moments/mean:output:0*
T0*&
_output_shapes
: 2-
+batch_normalization_25/moments/StopGradient
0batch_normalization_25/moments/SquaredDifferenceSquaredDifferenceconv2d_25/Relu:activations:04batch_normalization_25/moments/StopGradient:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 22
0batch_normalization_25/moments/SquaredDifferenceË
9batch_normalization_25/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2;
9batch_normalization_25/moments/variance/reduction_indices
'batch_normalization_25/moments/varianceMean4batch_normalization_25/moments/SquaredDifference:z:0Bbatch_normalization_25/moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(2)
'batch_normalization_25/moments/varianceÇ
&batch_normalization_25/moments/SqueezeSqueeze,batch_normalization_25/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2(
&batch_normalization_25/moments/SqueezeÏ
(batch_normalization_25/moments/Squeeze_1Squeeze0batch_normalization_25/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2*
(batch_normalization_25/moments/Squeeze_1æ
,batch_normalization_25/AssignMovingAvg/decayConst*C
_class9
75loc:@batch_normalization_25/AssignMovingAvg/534954990*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_25/AssignMovingAvg/decay
+batch_normalization_25/AssignMovingAvg/CastCast5batch_normalization_25/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*C
_class9
75loc:@batch_normalization_25/AssignMovingAvg/534954990*
_output_shapes
: 2-
+batch_normalization_25/AssignMovingAvg/CastÛ
5batch_normalization_25/AssignMovingAvg/ReadVariableOpReadVariableOp0batch_normalization_25_assignmovingavg_534954990*
_output_shapes
: *
dtype027
5batch_normalization_25/AssignMovingAvg/ReadVariableOp¹
*batch_normalization_25/AssignMovingAvg/subSub=batch_normalization_25/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_25/moments/Squeeze:output:0*
T0*C
_class9
75loc:@batch_normalization_25/AssignMovingAvg/534954990*
_output_shapes
: 2,
*batch_normalization_25/AssignMovingAvg/subª
*batch_normalization_25/AssignMovingAvg/mulMul.batch_normalization_25/AssignMovingAvg/sub:z:0/batch_normalization_25/AssignMovingAvg/Cast:y:0*
T0*C
_class9
75loc:@batch_normalization_25/AssignMovingAvg/534954990*
_output_shapes
: 2,
*batch_normalization_25/AssignMovingAvg/mul
:batch_normalization_25/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp0batch_normalization_25_assignmovingavg_534954990.batch_normalization_25/AssignMovingAvg/mul:z:06^batch_normalization_25/AssignMovingAvg/ReadVariableOp*C
_class9
75loc:@batch_normalization_25/AssignMovingAvg/534954990*
_output_shapes
 *
dtype02<
:batch_normalization_25/AssignMovingAvg/AssignSubVariableOpì
.batch_normalization_25/AssignMovingAvg_1/decayConst*E
_class;
97loc:@batch_normalization_25/AssignMovingAvg_1/534954997*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_25/AssignMovingAvg_1/decay
-batch_normalization_25/AssignMovingAvg_1/CastCast7batch_normalization_25/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*E
_class;
97loc:@batch_normalization_25/AssignMovingAvg_1/534954997*
_output_shapes
: 2/
-batch_normalization_25/AssignMovingAvg_1/Castá
7batch_normalization_25/AssignMovingAvg_1/ReadVariableOpReadVariableOp2batch_normalization_25_assignmovingavg_1_534954997*
_output_shapes
: *
dtype029
7batch_normalization_25/AssignMovingAvg_1/ReadVariableOpÃ
,batch_normalization_25/AssignMovingAvg_1/subSub?batch_normalization_25/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_25/moments/Squeeze_1:output:0*
T0*E
_class;
97loc:@batch_normalization_25/AssignMovingAvg_1/534954997*
_output_shapes
: 2.
,batch_normalization_25/AssignMovingAvg_1/sub´
,batch_normalization_25/AssignMovingAvg_1/mulMul0batch_normalization_25/AssignMovingAvg_1/sub:z:01batch_normalization_25/AssignMovingAvg_1/Cast:y:0*
T0*E
_class;
97loc:@batch_normalization_25/AssignMovingAvg_1/534954997*
_output_shapes
: 2.
,batch_normalization_25/AssignMovingAvg_1/mul
<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp2batch_normalization_25_assignmovingavg_1_5349549970batch_normalization_25/AssignMovingAvg_1/mul:z:08^batch_normalization_25/AssignMovingAvg_1/ReadVariableOp*E
_class;
97loc:@batch_normalization_25/AssignMovingAvg_1/534954997*
_output_shapes
 *
dtype02>
<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_25/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2(
&batch_normalization_25/batchnorm/add/yÞ
$batch_normalization_25/batchnorm/addAddV21batch_normalization_25/moments/Squeeze_1:output:0/batch_normalization_25/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_25/batchnorm/add¨
&batch_normalization_25/batchnorm/RsqrtRsqrt(batch_normalization_25/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_25/batchnorm/Rsqrtã
3batch_normalization_25/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_25_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_25/batchnorm/mul/ReadVariableOpá
$batch_normalization_25/batchnorm/mulMul*batch_normalization_25/batchnorm/Rsqrt:y:0;batch_normalization_25/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_25/batchnorm/mulÙ
&batch_normalization_25/batchnorm/mul_1Mulconv2d_25/Relu:activations:0(batch_normalization_25/batchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2(
&batch_normalization_25/batchnorm/mul_1×
&batch_normalization_25/batchnorm/mul_2Mul/batch_normalization_25/moments/Squeeze:output:0(batch_normalization_25/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_25/batchnorm/mul_2×
/batch_normalization_25/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_25_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_25/batchnorm/ReadVariableOpÝ
$batch_normalization_25/batchnorm/subSub7batch_normalization_25/batchnorm/ReadVariableOp:value:0*batch_normalization_25/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_25/batchnorm/subé
&batch_normalization_25/batchnorm/add_1AddV2*batch_normalization_25/batchnorm/mul_1:z:0(batch_normalization_25/batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2(
&batch_normalization_25/batchnorm/add_1á
max_pooling2d_17/MaxPoolMaxPool*batch_normalization_25/batchnorm/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_17/MaxPool}
dropout_25/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB 2      ô?2
dropout_25/dropout/Const·
dropout_25/dropout/MulMul!max_pooling2d_17/MaxPool:output:0!dropout_25/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_25/dropout/Mul
dropout_25/dropout/ShapeShape!max_pooling2d_17/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_25/dropout/ShapeÝ
/dropout_25/dropout/random_uniform/RandomUniformRandomUniform!dropout_25/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype021
/dropout_25/dropout/random_uniform/RandomUniform
!dropout_25/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB 2É?2#
!dropout_25/dropout/GreaterEqual/yò
dropout_25/dropout/GreaterEqualGreaterEqual8dropout_25/dropout/random_uniform/RandomUniform:output:0*dropout_25/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
dropout_25/dropout/GreaterEqual¨
dropout_25/dropout/CastCast#dropout_25/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_25/dropout/Cast®
dropout_25/dropout/Mul_1Muldropout_25/dropout/Mul:z:0dropout_25/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_25/dropout/Mul_1³
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_26/Conv2D/ReadVariableOpØ
conv2d_26/Conv2DConv2Ddropout_25/dropout/Mul_1:z:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_26/Conv2Dª
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp°
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_26/BiasAdd~
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_26/ReluÃ
5batch_normalization_26/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          27
5batch_normalization_26/moments/mean/reduction_indicesò
#batch_normalization_26/moments/meanMeanconv2d_26/Relu:activations:0>batch_normalization_26/moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(2%
#batch_normalization_26/moments/meanÉ
+batch_normalization_26/moments/StopGradientStopGradient,batch_normalization_26/moments/mean:output:0*
T0*&
_output_shapes
:@2-
+batch_normalization_26/moments/StopGradient
0batch_normalization_26/moments/SquaredDifferenceSquaredDifferenceconv2d_26/Relu:activations:04batch_normalization_26/moments/StopGradient:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@22
0batch_normalization_26/moments/SquaredDifferenceË
9batch_normalization_26/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2;
9batch_normalization_26/moments/variance/reduction_indices
'batch_normalization_26/moments/varianceMean4batch_normalization_26/moments/SquaredDifference:z:0Bbatch_normalization_26/moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(2)
'batch_normalization_26/moments/varianceÇ
&batch_normalization_26/moments/SqueezeSqueeze,batch_normalization_26/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2(
&batch_normalization_26/moments/SqueezeÏ
(batch_normalization_26/moments/Squeeze_1Squeeze0batch_normalization_26/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2*
(batch_normalization_26/moments/Squeeze_1æ
,batch_normalization_26/AssignMovingAvg/decayConst*C
_class9
75loc:@batch_normalization_26/AssignMovingAvg/534955040*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_26/AssignMovingAvg/decay
+batch_normalization_26/AssignMovingAvg/CastCast5batch_normalization_26/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*C
_class9
75loc:@batch_normalization_26/AssignMovingAvg/534955040*
_output_shapes
: 2-
+batch_normalization_26/AssignMovingAvg/CastÛ
5batch_normalization_26/AssignMovingAvg/ReadVariableOpReadVariableOp0batch_normalization_26_assignmovingavg_534955040*
_output_shapes
:@*
dtype027
5batch_normalization_26/AssignMovingAvg/ReadVariableOp¹
*batch_normalization_26/AssignMovingAvg/subSub=batch_normalization_26/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_26/moments/Squeeze:output:0*
T0*C
_class9
75loc:@batch_normalization_26/AssignMovingAvg/534955040*
_output_shapes
:@2,
*batch_normalization_26/AssignMovingAvg/subª
*batch_normalization_26/AssignMovingAvg/mulMul.batch_normalization_26/AssignMovingAvg/sub:z:0/batch_normalization_26/AssignMovingAvg/Cast:y:0*
T0*C
_class9
75loc:@batch_normalization_26/AssignMovingAvg/534955040*
_output_shapes
:@2,
*batch_normalization_26/AssignMovingAvg/mul
:batch_normalization_26/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp0batch_normalization_26_assignmovingavg_534955040.batch_normalization_26/AssignMovingAvg/mul:z:06^batch_normalization_26/AssignMovingAvg/ReadVariableOp*C
_class9
75loc:@batch_normalization_26/AssignMovingAvg/534955040*
_output_shapes
 *
dtype02<
:batch_normalization_26/AssignMovingAvg/AssignSubVariableOpì
.batch_normalization_26/AssignMovingAvg_1/decayConst*E
_class;
97loc:@batch_normalization_26/AssignMovingAvg_1/534955047*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_26/AssignMovingAvg_1/decay
-batch_normalization_26/AssignMovingAvg_1/CastCast7batch_normalization_26/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*E
_class;
97loc:@batch_normalization_26/AssignMovingAvg_1/534955047*
_output_shapes
: 2/
-batch_normalization_26/AssignMovingAvg_1/Castá
7batch_normalization_26/AssignMovingAvg_1/ReadVariableOpReadVariableOp2batch_normalization_26_assignmovingavg_1_534955047*
_output_shapes
:@*
dtype029
7batch_normalization_26/AssignMovingAvg_1/ReadVariableOpÃ
,batch_normalization_26/AssignMovingAvg_1/subSub?batch_normalization_26/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_26/moments/Squeeze_1:output:0*
T0*E
_class;
97loc:@batch_normalization_26/AssignMovingAvg_1/534955047*
_output_shapes
:@2.
,batch_normalization_26/AssignMovingAvg_1/sub´
,batch_normalization_26/AssignMovingAvg_1/mulMul0batch_normalization_26/AssignMovingAvg_1/sub:z:01batch_normalization_26/AssignMovingAvg_1/Cast:y:0*
T0*E
_class;
97loc:@batch_normalization_26/AssignMovingAvg_1/534955047*
_output_shapes
:@2.
,batch_normalization_26/AssignMovingAvg_1/mul
<batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp2batch_normalization_26_assignmovingavg_1_5349550470batch_normalization_26/AssignMovingAvg_1/mul:z:08^batch_normalization_26/AssignMovingAvg_1/ReadVariableOp*E
_class;
97loc:@batch_normalization_26/AssignMovingAvg_1/534955047*
_output_shapes
 *
dtype02>
<batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2(
&batch_normalization_26/batchnorm/add/yÞ
$batch_normalization_26/batchnorm/addAddV21batch_normalization_26/moments/Squeeze_1:output:0/batch_normalization_26/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2&
$batch_normalization_26/batchnorm/add¨
&batch_normalization_26/batchnorm/RsqrtRsqrt(batch_normalization_26/batchnorm/add:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_26/batchnorm/Rsqrtã
3batch_normalization_26/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_26_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3batch_normalization_26/batchnorm/mul/ReadVariableOpá
$batch_normalization_26/batchnorm/mulMul*batch_normalization_26/batchnorm/Rsqrt:y:0;batch_normalization_26/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2&
$batch_normalization_26/batchnorm/mulÙ
&batch_normalization_26/batchnorm/mul_1Mulconv2d_26/Relu:activations:0(batch_normalization_26/batchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&batch_normalization_26/batchnorm/mul_1×
&batch_normalization_26/batchnorm/mul_2Mul/batch_normalization_26/moments/Squeeze:output:0(batch_normalization_26/batchnorm/mul:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_26/batchnorm/mul_2×
/batch_normalization_26/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_26_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/batch_normalization_26/batchnorm/ReadVariableOpÝ
$batch_normalization_26/batchnorm/subSub7batch_normalization_26/batchnorm/ReadVariableOp:value:0*batch_normalization_26/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2&
$batch_normalization_26/batchnorm/subé
&batch_normalization_26/batchnorm/add_1AddV2*batch_normalization_26/batchnorm/mul_1:z:0(batch_normalization_26/batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&batch_normalization_26/batchnorm/add_1}
dropout_26/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB 2·mÛ¶mÛö?2
dropout_26/dropout/ConstÀ
dropout_26/dropout/MulMul*batch_normalization_26/batchnorm/add_1:z:0!dropout_26/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_26/dropout/Mul
dropout_26/dropout/ShapeShape*batch_normalization_26/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dropout_26/dropout/ShapeÝ
/dropout_26/dropout/random_uniform/RandomUniformRandomUniform!dropout_26/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype021
/dropout_26/dropout/random_uniform/RandomUniform
!dropout_26/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB 2433333Ó?2#
!dropout_26/dropout/GreaterEqual/yò
dropout_26/dropout/GreaterEqualGreaterEqual8dropout_26/dropout/random_uniform/RandomUniform:output:0*dropout_26/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
dropout_26/dropout/GreaterEqual¨
dropout_26/dropout/CastCast#dropout_26/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_26/dropout/Cast®
dropout_26/dropout/Mul_1Muldropout_26/dropout/Mul:z:0dropout_26/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_26/dropout/Mul_1s
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
flatten_8/Const
flatten_8/ReshapeReshapedropout_26/dropout/Mul_1:z:0flatten_8/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
flatten_8/Reshape¦
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	À
*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMulflatten_8/Reshape:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_8/MatMul¤
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_8/BiasAdd/ReadVariableOp¡
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_8/BiasAddà
IdentityIdentitydense_8/BiasAdd:output:0;^batch_normalization_24/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_25/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_26/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::2x
:batch_normalization_24/AssignMovingAvg/AssignSubVariableOp:batch_normalization_24/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_25/AssignMovingAvg/AssignSubVariableOp:batch_normalization_25/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_26/AssignMovingAvg/AssignSubVariableOp:batch_normalization_26/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOp:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_1
l
Ù	
I__inference_conv_net_9_layer_call_and_return_conditional_losses_534955169
input_1,
(conv2d_24_conv2d_readvariableop_resource-
)conv2d_24_biasadd_readvariableop_resource<
8batch_normalization_24_batchnorm_readvariableop_resource@
<batch_normalization_24_batchnorm_mul_readvariableop_resource>
:batch_normalization_24_batchnorm_readvariableop_1_resource>
:batch_normalization_24_batchnorm_readvariableop_2_resource,
(conv2d_25_conv2d_readvariableop_resource-
)conv2d_25_biasadd_readvariableop_resource<
8batch_normalization_25_batchnorm_readvariableop_resource@
<batch_normalization_25_batchnorm_mul_readvariableop_resource>
:batch_normalization_25_batchnorm_readvariableop_1_resource>
:batch_normalization_25_batchnorm_readvariableop_2_resource,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource<
8batch_normalization_26_batchnorm_readvariableop_resource@
<batch_normalization_26_batchnorm_mul_readvariableop_resource>
:batch_normalization_26_batchnorm_readvariableop_1_resource>
:batch_normalization_26_batchnorm_readvariableop_2_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource
identity³
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_24/Conv2D/ReadVariableOpÃ
conv2d_24/Conv2DConv2Dinput_1'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_24/Conv2Dª
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp°
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_24/BiasAdd~
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_24/Relu×
/batch_normalization_24/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_24_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_24/batchnorm/ReadVariableOp
&batch_normalization_24/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2(
&batch_normalization_24/batchnorm/add/yä
$batch_normalization_24/batchnorm/addAddV27batch_normalization_24/batchnorm/ReadVariableOp:value:0/batch_normalization_24/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_24/batchnorm/add¨
&batch_normalization_24/batchnorm/RsqrtRsqrt(batch_normalization_24/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_24/batchnorm/Rsqrtã
3batch_normalization_24/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_24_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_24/batchnorm/mul/ReadVariableOpá
$batch_normalization_24/batchnorm/mulMul*batch_normalization_24/batchnorm/Rsqrt:y:0;batch_normalization_24/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_24/batchnorm/mulÙ
&batch_normalization_24/batchnorm/mul_1Mulconv2d_24/Relu:activations:0(batch_normalization_24/batchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&batch_normalization_24/batchnorm/mul_1Ý
1batch_normalization_24/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_24_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype023
1batch_normalization_24/batchnorm/ReadVariableOp_1á
&batch_normalization_24/batchnorm/mul_2Mul9batch_normalization_24/batchnorm/ReadVariableOp_1:value:0(batch_normalization_24/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_24/batchnorm/mul_2Ý
1batch_normalization_24/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_24_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype023
1batch_normalization_24/batchnorm/ReadVariableOp_2ß
$batch_normalization_24/batchnorm/subSub9batch_normalization_24/batchnorm/ReadVariableOp_2:value:0*batch_normalization_24/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_24/batchnorm/subé
&batch_normalization_24/batchnorm/add_1AddV2*batch_normalization_24/batchnorm/mul_1:z:0(batch_normalization_24/batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&batch_normalization_24/batchnorm/add_1á
max_pooling2d_16/MaxPoolMaxPool*batch_normalization_24/batchnorm/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_16/MaxPool
dropout_24/IdentityIdentity!max_pooling2d_16/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_24/Identity³
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_25/Conv2D/ReadVariableOpØ
conv2d_25/Conv2DConv2Ddropout_24/Identity:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *
paddingVALID*
strides
2
conv2d_25/Conv2Dª
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_25/BiasAdd/ReadVariableOp°
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
conv2d_25/BiasAdd~
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
conv2d_25/Relu×
/batch_normalization_25/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_25_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_25/batchnorm/ReadVariableOp
&batch_normalization_25/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2(
&batch_normalization_25/batchnorm/add/yä
$batch_normalization_25/batchnorm/addAddV27batch_normalization_25/batchnorm/ReadVariableOp:value:0/batch_normalization_25/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_25/batchnorm/add¨
&batch_normalization_25/batchnorm/RsqrtRsqrt(batch_normalization_25/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_25/batchnorm/Rsqrtã
3batch_normalization_25/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_25_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_25/batchnorm/mul/ReadVariableOpá
$batch_normalization_25/batchnorm/mulMul*batch_normalization_25/batchnorm/Rsqrt:y:0;batch_normalization_25/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_25/batchnorm/mulÙ
&batch_normalization_25/batchnorm/mul_1Mulconv2d_25/Relu:activations:0(batch_normalization_25/batchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2(
&batch_normalization_25/batchnorm/mul_1Ý
1batch_normalization_25/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_25_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype023
1batch_normalization_25/batchnorm/ReadVariableOp_1á
&batch_normalization_25/batchnorm/mul_2Mul9batch_normalization_25/batchnorm/ReadVariableOp_1:value:0(batch_normalization_25/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_25/batchnorm/mul_2Ý
1batch_normalization_25/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_25_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype023
1batch_normalization_25/batchnorm/ReadVariableOp_2ß
$batch_normalization_25/batchnorm/subSub9batch_normalization_25/batchnorm/ReadVariableOp_2:value:0*batch_normalization_25/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_25/batchnorm/subé
&batch_normalization_25/batchnorm/add_1AddV2*batch_normalization_25/batchnorm/mul_1:z:0(batch_normalization_25/batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2(
&batch_normalization_25/batchnorm/add_1á
max_pooling2d_17/MaxPoolMaxPool*batch_normalization_25/batchnorm/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_17/MaxPool
dropout_25/IdentityIdentity!max_pooling2d_17/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_25/Identity³
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_26/Conv2D/ReadVariableOpØ
conv2d_26/Conv2DConv2Ddropout_25/Identity:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_26/Conv2Dª
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp°
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_26/BiasAdd~
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_26/Relu×
/batch_normalization_26/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_26_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/batch_normalization_26/batchnorm/ReadVariableOp
&batch_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2ü©ñÒMbP?2(
&batch_normalization_26/batchnorm/add/yä
$batch_normalization_26/batchnorm/addAddV27batch_normalization_26/batchnorm/ReadVariableOp:value:0/batch_normalization_26/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2&
$batch_normalization_26/batchnorm/add¨
&batch_normalization_26/batchnorm/RsqrtRsqrt(batch_normalization_26/batchnorm/add:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_26/batchnorm/Rsqrtã
3batch_normalization_26/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_26_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3batch_normalization_26/batchnorm/mul/ReadVariableOpá
$batch_normalization_26/batchnorm/mulMul*batch_normalization_26/batchnorm/Rsqrt:y:0;batch_normalization_26/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2&
$batch_normalization_26/batchnorm/mulÙ
&batch_normalization_26/batchnorm/mul_1Mulconv2d_26/Relu:activations:0(batch_normalization_26/batchnorm/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&batch_normalization_26/batchnorm/mul_1Ý
1batch_normalization_26/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_26_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype023
1batch_normalization_26/batchnorm/ReadVariableOp_1á
&batch_normalization_26/batchnorm/mul_2Mul9batch_normalization_26/batchnorm/ReadVariableOp_1:value:0(batch_normalization_26/batchnorm/mul:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_26/batchnorm/mul_2Ý
1batch_normalization_26/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_26_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype023
1batch_normalization_26/batchnorm/ReadVariableOp_2ß
$batch_normalization_26/batchnorm/subSub9batch_normalization_26/batchnorm/ReadVariableOp_2:value:0*batch_normalization_26/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2&
$batch_normalization_26/batchnorm/subé
&batch_normalization_26/batchnorm/add_1AddV2*batch_normalization_26/batchnorm/mul_1:z:0(batch_normalization_26/batchnorm/sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&batch_normalization_26/batchnorm/add_1
dropout_26/IdentityIdentity*batch_normalization_26/batchnorm/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_26/Identitys
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
flatten_8/Const
flatten_8/ReshapeReshapedropout_26/Identity:output:0flatten_8/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
flatten_8/Reshape¦
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	À
*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMulflatten_8/Reshape:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_8/MatMul¤
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_8/BiasAdd/ReadVariableOp¡
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_8/BiasAddl
IdentityIdentitydense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ  :::::::::::::::::::::X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_1
Á
J
.__inference_dropout_24_layer_call_fn_534955811

inputs
identityÔ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_24_layer_call_and_return_conditional_losses_5349542582
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
°
H__inference_conv2d_25_layer_call_and_return_conditional_losses_534955822

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ì
g
I__inference_dropout_25_layer_call_and_return_conditional_losses_534956016

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
°
H__inference_conv2d_25_layer_call_and_return_conditional_losses_534954282

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


-__inference_conv2d_24_layer_call_fn_534955616

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_conv2d_24_layer_call_and_return_conditional_losses_5349541312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ  ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
À
d
H__inference_flatten_8_layer_call_and_return_conditional_losses_534956247

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¥

'__inference_signature_wrapper_534954922
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity¢StatefulPartitionedCallÍ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *-
f(R&
$__inference__wrapped_model_5349536602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_1
ª
­
:__inference_batch_normalization_25_layer_call_fn_534955902

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *^
fYRW
U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5349539162
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*³
serving_default
C
input_18
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ  <
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:Ú×
â
	conv1
batch_norm1
max1
dropout1
	conv2
batch_norm2
max2
dropout2
		conv3

batch_norm3
dropout3
flatten
	dense
regularization_losses
trainable_variables
	variables
	keras_api

signatures
¤__call__
+¥&call_and_return_all_conditional_losses
¦_default_save_signature"ù
_tf_keras_modelß{"class_name": "ConvNet", "name": "conv_net_9", "trainable": true, "expects_training_arg": true, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ConvNet"}}
í


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
§__call__
+¨&call_and_return_all_conditional_losses"Æ	
_tf_keras_layer¬	{"class_name": "Conv2D", "name": "conv2d_24", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_24", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float64", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "SmallRandomNormal", "config": {}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [200, 32, 32, 3]}}
½	
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
trainable_variables
 	variables
!	keras_api
©__call__
+ª&call_and_return_all_conditional_losses"ç
_tf_keras_layerÍ{"class_name": "BatchNormalization", "name": "batch_normalization_24", "trainable": true, "expects_training_arg": true, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_24", "trainable": true, "dtype": "float64", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [200, 28, 28, 32]}}

"regularization_losses
#trainable_variables
$	variables
%	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"ò
_tf_keras_layerØ{"class_name": "MaxPooling2D", "name": "max_pooling2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_16", "trainable": true, "dtype": "float64", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
é
&regularization_losses
'trainable_variables
(	variables
)	keras_api
­__call__
+®&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_24", "trainable": true, "expects_training_arg": true, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_24", "trainable": true, "dtype": "float64", "rate": 0.1, "noise_shape": null, "seed": null}}
ñ


*kernel
+bias
,regularization_losses
-trainable_variables
.	variables
/	keras_api
¯__call__
+°&call_and_return_all_conditional_losses"Ê	
_tf_keras_layer°	{"class_name": "Conv2D", "name": "conv2d_25", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 14, 14, 32]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_25", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 14, 14, 32]}, "dtype": "float64", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "SmallRandomNormal", "config": {}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [200, 14, 14, 32]}}
½	
0axis
	1gamma
2beta
3moving_mean
4moving_variance
5regularization_losses
6trainable_variables
7	variables
8	keras_api
±__call__
+²&call_and_return_all_conditional_losses"ç
_tf_keras_layerÍ{"class_name": "BatchNormalization", "name": "batch_normalization_25", "trainable": true, "expects_training_arg": true, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_25", "trainable": true, "dtype": "float64", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [200, 10, 10, 32]}}

9regularization_losses
:trainable_variables
;	variables
<	keras_api
³__call__
+´&call_and_return_all_conditional_losses"ò
_tf_keras_layerØ{"class_name": "MaxPooling2D", "name": "max_pooling2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_17", "trainable": true, "dtype": "float64", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
é
=regularization_losses
>trainable_variables
?	variables
@	keras_api
µ__call__
+¶&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_25", "trainable": true, "expects_training_arg": true, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_25", "trainable": true, "dtype": "float64", "rate": 0.2, "noise_shape": null, "seed": null}}
ë


Akernel
Bbias
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
·__call__
+¸&call_and_return_all_conditional_losses"Ä	
_tf_keras_layerª	{"class_name": "Conv2D", "name": "conv2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 5, 32]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_26", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 5, 32]}, "dtype": "float64", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "SmallRandomNormal", "config": {}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [200, 5, 5, 32]}}
»	
Gaxis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
¹__call__
+º&call_and_return_all_conditional_losses"å
_tf_keras_layerË{"class_name": "BatchNormalization", "name": "batch_normalization_26", "trainable": true, "expects_training_arg": true, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_26", "trainable": true, "dtype": "float64", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [200, 3, 3, 64]}}
ù
Pregularization_losses
Qtrainable_variables
R	variables
S	keras_api
»__call__
+¼&call_and_return_all_conditional_losses"è
_tf_keras_layerÎ{"class_name": "Dropout", "name": "dropout_26", "trainable": true, "expects_training_arg": true, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_26", "trainable": true, "dtype": "float64", "rate": 0.30000000000000004, "noise_shape": null, "seed": null}}
è
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
½__call__
+¾&call_and_return_all_conditional_losses"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_8", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_8", "trainable": true, "dtype": "float64", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
í

Xkernel
Ybias
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
¿__call__
+À&call_and_return_all_conditional_losses"Æ
_tf_keras_layer¬{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float64", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "SmallRandomNormal", "config": {}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 576}}}, "build_input_shape": {"class_name": "TensorShape", "items": [200, 576]}}
 "
trackable_list_wrapper

0
1
2
3
*4
+5
16
27
A8
B9
H10
I11
X12
Y13"
trackable_list_wrapper
¶
0
1
2
3
4
5
*6
+7
18
29
310
411
A12
B13
H14
I15
J16
K17
X18
Y19"
trackable_list_wrapper
Î
^non_trainable_variables
_layer_metrics
regularization_losses

`layers
alayer_regularization_losses
bmetrics
trainable_variables
	variables
¤__call__
¦_default_save_signature
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
-
Áserving_default"
signature_map
5:3 2conv_net_9/conv2d_24/kernel
':% 2conv_net_9/conv2d_24/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
cnon_trainable_variables
dlayer_metrics
regularization_losses

elayers
flayer_regularization_losses
gmetrics
trainable_variables
	variables
§__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
5:3 2'conv_net_9/batch_normalization_24/gamma
4:2 2&conv_net_9/batch_normalization_24/beta
=:;  (2-conv_net_9/batch_normalization_24/moving_mean
A:?  (21conv_net_9/batch_normalization_24/moving_variance
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
°
hnon_trainable_variables
ilayer_metrics
regularization_losses

jlayers
klayer_regularization_losses
lmetrics
trainable_variables
 	variables
©__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
mnon_trainable_variables
nlayer_metrics
"regularization_losses

olayers
player_regularization_losses
qmetrics
#trainable_variables
$	variables
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
rnon_trainable_variables
slayer_metrics
&regularization_losses

tlayers
ulayer_regularization_losses
vmetrics
'trainable_variables
(	variables
­__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
5:3  2conv_net_9/conv2d_25/kernel
':% 2conv_net_9/conv2d_25/bias
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
°
wnon_trainable_variables
xlayer_metrics
,regularization_losses

ylayers
zlayer_regularization_losses
{metrics
-trainable_variables
.	variables
¯__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
5:3 2'conv_net_9/batch_normalization_25/gamma
4:2 2&conv_net_9/batch_normalization_25/beta
=:;  (2-conv_net_9/batch_normalization_25/moving_mean
A:?  (21conv_net_9/batch_normalization_25/moving_variance
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
<
10
21
32
43"
trackable_list_wrapper
±
|non_trainable_variables
}layer_metrics
5regularization_losses

~layers
layer_regularization_losses
metrics
6trainable_variables
7	variables
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
9regularization_losses
layers
 layer_regularization_losses
metrics
:trainable_variables
;	variables
³__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
=regularization_losses
layers
 layer_regularization_losses
metrics
>trainable_variables
?	variables
µ__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
5:3 @2conv_net_9/conv2d_26/kernel
':%@2conv_net_9/conv2d_26/bias
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
Cregularization_losses
layers
 layer_regularization_losses
metrics
Dtrainable_variables
E	variables
·__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
5:3@2'conv_net_9/batch_normalization_26/gamma
4:2@2&conv_net_9/batch_normalization_26/beta
=:;@ (2-conv_net_9/batch_normalization_26/moving_mean
A:?@ (21conv_net_9/batch_normalization_26/moving_variance
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
<
H0
I1
J2
K3"
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
Lregularization_losses
layers
 layer_regularization_losses
metrics
Mtrainable_variables
N	variables
¹__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
Pregularization_losses
layers
 layer_regularization_losses
metrics
Qtrainable_variables
R	variables
»__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
Tregularization_losses
layers
 layer_regularization_losses
metrics
Utrainable_variables
V	variables
½__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses"
_generic_user_object
,:*	À
2conv_net_9/dense_8/kernel
%:#
2conv_net_9/dense_8/bias
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
µ
non_trainable_variables
 layer_metrics
Zregularization_losses
¡layers
 ¢layer_regularization_losses
£metrics
[trainable_variables
\	variables
¿__call__
+À&call_and_return_all_conditional_losses
'À"call_and_return_conditional_losses"
_generic_user_object
J
0
1
32
43
J4
K5"
trackable_list_wrapper
 "
trackable_dict_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
õ2ò
.__inference_conv_net_9_layer_call_fn_534955259
.__inference_conv_net_9_layer_call_fn_534955214
.__inference_conv_net_9_layer_call_fn_534955551
.__inference_conv_net_9_layer_call_fn_534955596¯
¦²¢
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
á2Þ
I__inference_conv_net_9_layer_call_and_return_conditional_losses_534955083
I__inference_conv_net_9_layer_call_and_return_conditional_losses_534955506
I__inference_conv_net_9_layer_call_and_return_conditional_losses_534955420
I__inference_conv_net_9_layer_call_and_return_conditional_losses_534955169¯
¦²¢
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
$__inference__wrapped_model_534953660¾
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ  
×2Ô
-__inference_conv2d_24_layer_call_fn_534955616¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_conv2d_24_layer_call_and_return_conditional_losses_534955607¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ª2§
:__inference_batch_normalization_24_layer_call_fn_534955700
:__inference_batch_normalization_24_layer_call_fn_534955771
:__inference_batch_normalization_24_layer_call_fn_534955784
:__inference_batch_normalization_24_layer_call_fn_534955687´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_534955758
U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_534955674
U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_534955654
U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_534955738´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
4__inference_max_pooling2d_16_layer_call_fn_534953816à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
·2´
O__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_534953810à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
.__inference_dropout_24_layer_call_fn_534955811
.__inference_dropout_24_layer_call_fn_534955806´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ð2Í
I__inference_dropout_24_layer_call_and_return_conditional_losses_534955796
I__inference_dropout_24_layer_call_and_return_conditional_losses_534955801´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
×2Ô
-__inference_conv2d_25_layer_call_fn_534955831¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_conv2d_25_layer_call_and_return_conditional_losses_534955822¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ª2§
:__inference_batch_normalization_25_layer_call_fn_534955999
:__inference_batch_normalization_25_layer_call_fn_534955915
:__inference_batch_normalization_25_layer_call_fn_534955902
:__inference_batch_normalization_25_layer_call_fn_534955986´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_534955973
U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_534955953
U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_534955869
U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_534955889´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
4__inference_max_pooling2d_17_layer_call_fn_534953972à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
·2´
O__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_534953966à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
.__inference_dropout_25_layer_call_fn_534956026
.__inference_dropout_25_layer_call_fn_534956021´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ð2Í
I__inference_dropout_25_layer_call_and_return_conditional_losses_534956016
I__inference_dropout_25_layer_call_and_return_conditional_losses_534956011´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
×2Ô
-__inference_conv2d_26_layer_call_fn_534956046¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_conv2d_26_layer_call_and_return_conditional_losses_534956037¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ª2§
:__inference_batch_normalization_26_layer_call_fn_534956214
:__inference_batch_normalization_26_layer_call_fn_534956130
:__inference_batch_normalization_26_layer_call_fn_534956117
:__inference_batch_normalization_26_layer_call_fn_534956201´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_534956188
U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_534956104
U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_534956084
U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_534956168´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
.__inference_dropout_26_layer_call_fn_534956236
.__inference_dropout_26_layer_call_fn_534956241´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ð2Í
I__inference_dropout_26_layer_call_and_return_conditional_losses_534956226
I__inference_dropout_26_layer_call_and_return_conditional_losses_534956231´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
×2Ô
-__inference_flatten_8_layer_call_fn_534956252¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_flatten_8_layer_call_and_return_conditional_losses_534956247¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_8_layer_call_fn_534956271¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_8_layer_call_and_return_conditional_losses_534956262¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
6B4
'__inference_signature_wrapper_534954922input_1®
$__inference__wrapped_model_534953660*+4132ABKHJIXY8¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ  
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
Ë
U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_534955654r;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Ë
U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_534955674r;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 ð
U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_534955738M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ð
U__inference_batch_normalization_24_layer_call_and_return_conditional_losses_534955758M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 £
:__inference_batch_normalization_24_layer_call_fn_534955687e;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª " ÿÿÿÿÿÿÿÿÿ £
:__inference_batch_normalization_24_layer_call_fn_534955700e;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª " ÿÿÿÿÿÿÿÿÿ È
:__inference_batch_normalization_24_layer_call_fn_534955771M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ È
:__inference_batch_normalization_24_layer_call_fn_534955784M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ð
U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5349558693412M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ð
U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_5349558894132M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ë
U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_534955953r3412;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ

 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ

 
 Ë
U__inference_batch_normalization_25_layer_call_and_return_conditional_losses_534955973r4132;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ

 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ

 
 È
:__inference_batch_normalization_25_layer_call_fn_5349559023412M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ È
:__inference_batch_normalization_25_layer_call_fn_5349559154132M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ £
:__inference_batch_normalization_25_layer_call_fn_534955986e3412;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ

 
p
ª " ÿÿÿÿÿÿÿÿÿ

 £
:__inference_batch_normalization_25_layer_call_fn_534955999e4132;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ

 
p 
ª " ÿÿÿÿÿÿÿÿÿ

 Ë
U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_534956084rJKHI;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Ë
U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_534956104rKHJI;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 ð
U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_534956168JKHIM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ð
U__inference_batch_normalization_26_layer_call_and_return_conditional_losses_534956188KHJIM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 £
:__inference_batch_normalization_26_layer_call_fn_534956117eJKHI;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@£
:__inference_batch_normalization_26_layer_call_fn_534956130eKHJI;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@È
:__inference_batch_normalization_26_layer_call_fn_534956201JKHIM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@È
:__inference_batch_normalization_26_layer_call_fn_534956214KHJIM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¸
H__inference_conv2d_24_layer_call_and_return_conditional_losses_534955607l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
-__inference_conv2d_24_layer_call_fn_534955616_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ ¸
H__inference_conv2d_25_layer_call_and_return_conditional_losses_534955822l*+7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ

 
 
-__inference_conv2d_25_layer_call_fn_534955831_*+7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ

 ¸
H__inference_conv2d_26_layer_call_and_return_conditional_losses_534956037lAB7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
-__inference_conv2d_26_layer_call_fn_534956046_AB7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ@È
I__inference_conv_net_9_layer_call_and_return_conditional_losses_534955083{*+3412ABJKHIXY<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ  
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 È
I__inference_conv_net_9_layer_call_and_return_conditional_losses_534955169{*+4132ABKHJIXY<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ  
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Â
I__inference_conv_net_9_layer_call_and_return_conditional_losses_534955420u*+3412ABJKHIXY6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿ  
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Â
I__inference_conv_net_9_layer_call_and_return_conditional_losses_534955506u*+4132ABKHJIXY6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿ  
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

  
.__inference_conv_net_9_layer_call_fn_534955214n*+3412ABJKHIXY<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ  
p
ª "ÿÿÿÿÿÿÿÿÿ
 
.__inference_conv_net_9_layer_call_fn_534955259n*+4132ABKHJIXY<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ  
p 
ª "ÿÿÿÿÿÿÿÿÿ

.__inference_conv_net_9_layer_call_fn_534955551h*+3412ABJKHIXY6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿ  
p
ª "ÿÿÿÿÿÿÿÿÿ

.__inference_conv_net_9_layer_call_fn_534955596h*+4132ABKHJIXY6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿ  
p 
ª "ÿÿÿÿÿÿÿÿÿ
§
F__inference_dense_8_layer_call_and_return_conditional_losses_534956262]XY0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
+__inference_dense_8_layer_call_fn_534956271PXY0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ
ª "ÿÿÿÿÿÿÿÿÿ
¹
I__inference_dropout_24_layer_call_and_return_conditional_losses_534955796l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 ¹
I__inference_dropout_24_layer_call_and_return_conditional_losses_534955801l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
.__inference_dropout_24_layer_call_fn_534955806_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª " ÿÿÿÿÿÿÿÿÿ 
.__inference_dropout_24_layer_call_fn_534955811_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª " ÿÿÿÿÿÿÿÿÿ ¹
I__inference_dropout_25_layer_call_and_return_conditional_losses_534956011l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 ¹
I__inference_dropout_25_layer_call_and_return_conditional_losses_534956016l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
.__inference_dropout_25_layer_call_fn_534956021_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª " ÿÿÿÿÿÿÿÿÿ 
.__inference_dropout_25_layer_call_fn_534956026_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª " ÿÿÿÿÿÿÿÿÿ ¹
I__inference_dropout_26_layer_call_and_return_conditional_losses_534956226l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 ¹
I__inference_dropout_26_layer_call_and_return_conditional_losses_534956231l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
.__inference_dropout_26_layer_call_fn_534956236_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@
.__inference_dropout_26_layer_call_fn_534956241_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@­
H__inference_flatten_8_layer_call_and_return_conditional_losses_534956247a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÀ
 
-__inference_flatten_8_layer_call_fn_534956252T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿÀò
O__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_534953810R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ê
4__inference_max_pooling2d_16_layer_call_fn_534953816R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿò
O__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_534953966R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ê
4__inference_max_pooling2d_17_layer_call_fn_534953972R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¼
'__inference_signature_wrapper_534954922*+4132ABKHJIXYC¢@
¢ 
9ª6
4
input_1)&
input_1ÿÿÿÿÿÿÿÿÿ  "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
