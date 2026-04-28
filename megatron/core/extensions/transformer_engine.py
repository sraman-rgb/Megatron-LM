# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses
import enum
import inspect
import io
import os
import pickle
import warnings
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, cast

import torch
import torch.nn.functional as F
from packaging.version import Version as PkgVersion
from torch import Tensor
from torch.nn.parameter import Parameter
from typing_extensions import override

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.enums import Fp4Recipe, Fp8Recipe
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_amax_reduction_group,
    get_context_parallel_group,
    get_hierarchical_context_parallel_groups,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
    model_parallel_is_initialized,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.quantization.quant_config import QuantizationConfig
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    set_tensor_model_parallel_attributes,
)
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
    get_expert_parallel_rng_tracker_name,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.torch_norm import LayerNormInterface
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    is_layer_window_attention,
    make_sharded_tensors_for_checkpoint,
)
from megatron.core.typed_torch import copy_signature
from megatron.core.utils import (
    get_pg_rank,
    get_pg_size,
    get_te_version,
    get_tensor_model_parallel_group_if_none,
    is_te_min_version,
    is_torch_min_version,
)

try:
    import transformer_engine as te
    from transformer_engine.pytorch.cpu_offload import is_cpu_offload_enabled, mark_activation_offload
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, fp8_autocast
    from transformer_engine.pytorch.ops._common import is_quantized_tensor as is_te_quantized_tensor

    HAVE_TE = True
except ImportError:
    if TYPE_CHECKING:
        # For type checking, treat transformer_engine as always available.
        import transformer_engine as te
        from transformer_engine.pytorch.cpu_offload import is_cpu_offload_enabled, mark_activation_offload
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, fp8_autocast
        from transformer_engine.pytorch.ops._common import is_quantized_tensor as is_te_quantized_tensor

        HAVE_TE = True
    else:
        from unittest.mock import MagicMock

        te = MagicMock()
        HAVE_TE = False

_TE_CONFIG_TYPE_KEY = "transformer_engine_config_type"


class TransformerEngineConfigType(enum.Enum):
    """Configuration object types in config dictionary"""

    TEQuantizationParams = "TEQuantizationParams"


@dataclasses.dataclass
class TEQuantizationRecipe:
    """Class to capture options for opening an autocast context in forward"""

    fp8_quantization_recipe: Optional[Fp8Recipe] = None
    """
    An FP8 quantization override if the module should use FP8.
    If no FP8 or FP4 quantization is configured, the recipe is execution
    in high-precision (BF16).
    """
    fp4_quantization_recipe: Optional[Fp4Recipe] = None
    """
    An FP4 quantization override if the module should use FP4.
    If no FP8 or FP4 quantization is configured, the recipe is execution
    in high-precision (BF16).
    """
    custom_recipe_factory: Optional[str] = None
    """The path to a custom recipe factory if a custom Fp4 or Fp8 recipe is configured"""
    fp8_format: str = "e4m3"
    """A format to select from an FP8Recipe"""
    override_quantized_autocast: bool = True
    """
    If the quantization autocast context for a targeted module is enabled,
    whether to override it and change (or disable) the quantization recipe.
    """
    override_nonquantized_autocast: bool = False
    """
    If the quantization autocast context for a targeted module is not enabled,
    whether to override it and enable a quantization recipe.
    """
    tp_only_amax_red: bool = False
    """
    If an amax reduction is applicable, such as in per-tensor quantization recipe,
    whether to reduce only along TP groups.
    """

    @classmethod
    def parse_from_config(cls, quant_config: Dict[Any, Any]) -> "TEQuantizationRecipe":
        """
        Parse config from quantization dictionary.
        """
        kwargs = {}
        class_keys = cls.get_config_keys()
        for field in class_keys:
            if field in quant_config:
                kwargs[field] = quant_config[field]
        for field in quant_config:
            if field not in class_keys:
                raise ValueError(f"Field '{field}' not valid for this configuration.")
        instance = TEQuantizationRecipe(**kwargs)
        if instance.fp8_quantization_recipe == Fp8Recipe.delayed:
            raise ValueError("Delayed scaling not in scope of te per-module quantization config.")
        if (
            instance.fp8_quantization_recipe is not None
            and instance.fp4_quantization_recipe is not None
        ):
            raise ValueError("fp8 and fp4 quantization settings are mutually exclusive.")
        if (
            instance.fp8_quantization_recipe == Fp8Recipe.custom
            or instance.fp4_quantization_recipe == Fp4Recipe.custom
        ):
            if instance.custom_recipe_factory is None:
                raise ValueError("custom fp8 or fp4 recipe requires custom_recipe_factory")
        return instance

    @classmethod
    def get_config_keys(cls) -> Set[str]:
        """Get expected keys from the dataclass fields."""
        return {field.name for field in dataclasses.fields(cls)}


@dataclasses.dataclass
class TEQuantizationParams:
    """Class to capture precision options for training and evaluation."""

    training_recipe: TEQuantizationRecipe
    """Precision override for when self.training is True"""
    evaluation_recipe: Optional[TEQuantizationRecipe]
    """
    Precision override for when self.training is False.
    If None, training_recipe is used.
    """

    @staticmethod
    def parse_from_config(quant_config: QuantizationConfig) -> "TEQuantizationParams":
        """Parses quantization config for a layer or throw an error."""
        config = quant_config.config
        try:
            config_type = TransformerEngineConfigType(config[_TE_CONFIG_TYPE_KEY])
        except KeyError:
            raise ValueError(
                f"TransformerEngine config dictionary must have '{_TE_CONFIG_TYPE_KEY}' key."
            )
        except ValueError:
            raise ValueError(f"Unsupported config type '{config[_TE_CONFIG_TYPE_KEY]}'.")

        if config_type == TransformerEngineConfigType.TEQuantizationParams:
            if 'training_recipe' not in config.keys():
                raise ValueError(
                    "TransformerEngine config dictionary must have 'training_recipe' key"
                )
            training_recipe = TEQuantizationRecipe.parse_from_config(config['training_recipe'])
            if 'evaluation_recipe' not in config.keys():
                evaluation_recipe = None
                assert len(config.keys()) == 2
            else:
                evaluation_recipe = TEQuantizationRecipe.parse_from_config(
                    config['evaluation_recipe']
                )
                assert len(config.keys()) == 3
            return TEQuantizationParams(
                training_recipe=training_recipe, evaluation_recipe=evaluation_recipe
            )
        else:
            raise NotImplementedError(f"Unhandled configuration type {config_type}")


def _get_fp8_autocast_for_quant_recipe(qrecipe: TEQuantizationRecipe):
    if FP8GlobalStateManager.is_fp8_enabled():
        if not qrecipe.override_quantized_autocast:
            return nullcontext()
    else:
        if not qrecipe.override_nonquantized_autocast:
            return nullcontext()

    if qrecipe.fp8_quantization_recipe is None and qrecipe.fp4_quantization_recipe is None:
        # Force BF16 for this layer and override autocast
        return fp8_autocast(enabled=False)
    else:
        amax_group = None
        if model_parallel_is_initialized():
            amax_group = get_amax_reduction_group(
                with_context_parallel=True, tp_only_amax_red=qrecipe.tp_only_amax_red
            )
        if (
            qrecipe.fp8_quantization_recipe == Fp8Recipe.custom
            or qrecipe.fp4_quantization_recipe == Fp4Recipe.custom
        ):
            from megatron.core.fp8_utils import _get_custom_recipe

            assert qrecipe.custom_recipe_factory is not None
            quant_recipe = _get_custom_recipe(qrecipe.custom_recipe_factory)
        elif qrecipe.fp8_quantization_recipe is not None:
            if qrecipe.fp8_format == "e4m3":
                fp8_format = te.common.recipe.Format.E4M3
            elif qrecipe.fp8_format == "hybrid":
                fp8_format = te.common.recipe.Format.HYBRID
            else:
                raise ValueError(f"Unhandled fp8_format {qrecipe.fp8_format}")

            if qrecipe.fp8_quantization_recipe == Fp8Recipe.tensorwise:
                quant_recipe = te.common.recipe.Float8CurrentScaling(fp8_format=fp8_format)
            elif qrecipe.fp8_quantization_recipe == Fp8Recipe.blockwise:
                quant_recipe = te.common.recipe.Float8BlockScaling(fp8_format=fp8_format)
            elif qrecipe.fp8_quantization_recipe == Fp8Recipe.mxfp8:
                quant_recipe = te.common.recipe.MXFP8BlockScaling(fp8_format=fp8_format)
            else:
                raise ValueError(f"Unhandled fp8 recipe: {qrecipe.fp8_quantization_recipe}")
        else:
            # Fp4 configured.
            if qrecipe.fp4_quantization_recipe == Fp4Recipe.nvfp4:
                quant_recipe = te.common.recipe.NVFP4BlockScaling()
            else:
                raise ValueError(f"Unhandled fp4 recipe: {qrecipe.fp8_quantization_recipe}")

        return fp8_autocast(enabled=True, fp8_recipe=quant_recipe, fp8_group=amax_group)


def _get_fp8_autocast_for_quant_params(qparams: TEQuantizationParams | None, training: bool):
    if qparams is None:
        return nullcontext()
    elif not training and qparams.evaluation_recipe is not None:
        return _get_fp8_autocast_for_quant_recipe(qparams.evaluation_recipe)
    else:
        return _get_fp8_autocast_for_quant_recipe(qparams.training_recipe)


def _get_should_context_be_quantized_recipe(
    qrecipe: TEQuantizationRecipe, is_original_context_quantized: bool
):
    if is_original_context_quantized:
        if not qrecipe.override_quantized_autocast:
            return is_original_context_quantized
    else:
        if not qrecipe.override_nonquantized_autocast:
            return is_original_context_quantized
    if qrecipe.fp8_quantization_recipe is None and qrecipe.fp4_quantization_recipe is None:
        # Force BF16 for this layer and override autocast
        return False
    else:
        return True


def _get_should_context_be_quantized_params(
    qparams: TEQuantizationParams | None, training: bool, is_context_quantized: bool
):
    if qparams is None:
        return is_context_quantized
    elif not training and qparams.evaluation_recipe is not None:
        return _get_should_context_be_quantized_recipe(
            qparams.evaluation_recipe, is_context_quantized
        )
    else:
        return _get_should_context_be_quantized_recipe(
            qparams.training_recipe, is_context_quantized
        )


def _get_extra_te_kwargs(config: TransformerConfig):
    extra_transformer_engine_kwargs = {"params_dtype": config.params_dtype}

    if is_te_min_version("0.12.0"):
        if config.use_cpu_initialization:
            extra_transformer_engine_kwargs["device"] = "cpu"
        elif config.init_model_with_meta_device:
            extra_transformer_engine_kwargs["device"] = "meta"
        else:
            extra_transformer_engine_kwargs["device"] = torch.cuda.current_device()
    return extra_transformer_engine_kwargs


def condition_init_method(config, init_method):
    """Condition TE init_method on config.perform_initialization."""
    return init_method if config.perform_initialization else (lambda w: None)


def split_te_layernorm_column_parallel_linear(
    fused_layer,
    config,
    init_method: Optional[callable] = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
):
    """
    Split a TELayerNormColumnParallelLinear into separate TENorm and TEColumnParallelLinear layers.

    Args:
        fused_layer: The fused TELayerNormColumnParallelLinear layer to split
        config: TransformerConfig to use for creating the new layers
        init_method: Initialization method for the linear layer (optional)
        tp_group: Tensor parallel group (optional)

    Returns:
        A tuple of (TENorm, TEColumnParallelLinear) with weights copied from the fused layer
    """

    # Extract dimensions from the fused layer
    in_features = fused_layer.in_features
    out_features = fused_layer.out_features * fused_layer.tp_size

    # Create the norm layer
    norm_layer = TENorm(config=config, hidden_size=in_features, eps=fused_layer.eps)

    with torch.no_grad():
        # Copy layer norm weight
        norm_layer.weight.copy_(fused_layer.layer_norm_weight)

        # Copy layer norm bias if it exists
        if hasattr(norm_layer, 'bias') and hasattr(fused_layer, 'layer_norm_bias'):
            if fused_layer.layer_norm_bias is not None:
                norm_layer.bias.copy_(fused_layer.layer_norm_bias)

    # Create the column parallel linear layer
    linear_layer = TEColumnParallelLinear(
        input_size=in_features,
        output_size=out_features,
        config=config,
        init_method=init_method or (lambda x: None),  # Dummy init since we'll copy weights
        gather_output=False,
        bias=fused_layer.use_bias,
        skip_bias_add=fused_layer.te_return_bias,
        is_expert=False,
        tp_comm_buffer_name=fused_layer.ub_name,
        tp_group=tp_group or fused_layer.tp_group,
    )

    with torch.no_grad():
        # Copy weight
        linear_layer.weight.copy_(fused_layer.weight)

        # Copy bias if it exists
        if fused_layer.use_bias and hasattr(fused_layer, 'bias'):
            linear_layer.bias.copy_(fused_layer.bias)

    # TODO(Peter): Do we need this
    # Copy FP8 metadata if applicable
    if hasattr(fused_layer, 'fp8_meta') and fused_layer.fp8_meta is not None:
        if hasattr(linear_layer, 'fp8_meta'):
            # Copy FP8 scaling factors and other metadata
            for key in fused_layer.fp8_meta:
                if key in linear_layer.fp8_meta:
                    if isinstance(fused_layer.fp8_meta[key], dict):
                        for subkey in fused_layer.fp8_meta[key]:
                            if subkey in linear_layer.fp8_meta[key]:
                                linear_layer.fp8_meta[key][subkey] = fused_layer.fp8_meta[key][
                                    subkey
                                ]
                    else:
                        linear_layer.fp8_meta[key] = fused_layer.fp8_meta[key]

    # Set the same configuration flags
    linear_layer.sequence_parallel = fused_layer.sequence_parallel
    linear_layer.is_first_microbatch = fused_layer.is_first_microbatch
    linear_layer.disable_parameter_transpose_cache = fused_layer.disable_parameter_transpose_cache

    return norm_layer, linear_layer


if HAVE_TE and is_te_min_version("1.13.0"):

    class TEActivationOp:
        """
        A conditional wrapper to initialize an instance of Transformer-Engine's activation
        function operators (e.g. Silu, SwiGLU, etc)
        """

        def __new__(cls, config: TransformerConfig):

            layer_type = None
            if config.gated_linear_unit:
                if config.activation_func == F.silu:
                    layer_type = te.pytorch.ops.SwiGLU
                elif config.activation_func == F.gelu:
                    layer_type = te.pytorch.ops.GEGLU
                elif config.activation_func == F.silu:
                    layer_type = te.pytorch.ops.ReGLU
            else:
                if config.activation_func == F.gelu:
                    layer_type = te.pytorch.ops.GELU
                elif config.activation_func == F.silu:
                    layer_type = te.pytorch.ops.ReLU
            if layer_type is None:
                raise Exception(
                    'Only SwiGLU, GEGLU, ReGLU, GELU, ReLU are supported by '
                    'transformer engine. Please set use_te_activation_func=False'
                )
            activation_func_kwargs = {}
            if config.activation_func_fp8_input_store:
                activation_func_kwargs["cache_quantized_input"] = True
            layer = layer_type(**activation_func_kwargs)
            return layer

else:
    TEActivationOp = None


if HAVE_TE and is_te_min_version("1.13.0"):

    class TEFusedResidualRMSNorm(te.pytorch.RMSNorm):
        """
        RMSNorm with fused residual output for Megatron Core.

        Inherits from te.pytorch.RMSNorm to maintain all parameter management,
        checkpoint compatibility, and Megatron-specific features. Creates a fused
        implementation using TE's ops API that shares the base class parameters.

        The fused implementation uses:
        - MakeExtraOutput: Forks the residual connection
        - RMSNorm: Normalizes the main path

        Forward pass returns: (normalized_output, residual)
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Fused implementation (stored in tuple to avoid submodule registration)
            self._fused_impl: Optional[Tuple[te.pytorch.ops.Sequential]] = None

        def _make_fused_impl(self) -> te.pytorch.ops.Sequential:
            """
            Construct fused ops pipeline that shares parameters with base RMSNorm.

            Creates MakeExtraOutput + RMSNorm ops, where the RMSNorm op shares
            the weight parameter with self.weight from the base class.
            """

            fused_impl = te.pytorch.ops.Sequential()

            # Op 1: MakeExtraOutput - forks the residual
            fused_impl.append(te.pytorch.ops.MakeExtraOutput())

            # Op 2: RMSNorm - shares weight parameter with self
            kwargs = {
                "eps": self.eps,
                "device": "meta",  # Already initialized
                "dtype": self.weight.dtype,
                "zero_centered_gamma": self.zero_centered_gamma,
            }

            # Add sm_margin if available (TE 2.5+)
            if hasattr(self, '_sm_margins'):
                kwargs["sm_margin"] = self._sm_margins

            rmsnorm_op = te.pytorch.ops.RMSNorm(self.weight.shape, **kwargs)

            rmsnorm_op.weight = self.weight

            fused_impl.append(rmsnorm_op)

            self._register_hooks_on_fused_impl(fused_impl)

            return fused_impl

        def _register_hooks_on_fused_impl(self, fused_impl: torch.nn.Module) -> None:

            forward_pre_hooks = []
            forward_post_hooks = []
            backward_pre_hooks = []
            backward_post_hooks = []

            for submodule in self.modules():
                for hook_id, hook in submodule._forward_pre_hooks.items():
                    with_kwargs = hook_id in submodule._forward_pre_hooks_with_kwargs
                    forward_pre_hooks.append((submodule, hook, with_kwargs))
                for hook_id, hook in submodule._forward_hooks.items():
                    with_kwargs = hook_id in submodule._forward_hooks_with_kwargs
                    forward_post_hooks.append((submodule, hook, with_kwargs))
                for hook in submodule._backward_pre_hooks.values():
                    backward_pre_hooks.append((submodule, hook))
                for hook in submodule._backward_hooks.values():
                    backward_post_hooks.append((submodule, hook))

            # Pre-forward hooks
            # Note: DDP pre-forward hooks are safe since they do not
            # interact with input tensor.
            if forward_pre_hooks:
                from megatron.core.distributed import distributed_data_parallel

                if any(
                    inspect.getmodule(hook) != distributed_data_parallel
                    for _, hook, _ in forward_pre_hooks
                ):
                    warnings.warn(
                        "TEFusedResidualRMSNorm module has a submodule with a pre-forward hook. "
                        "TEFusedResidualRMSNorm module does not expose intermediate tensors, "
                        "so the hook may have incorrect behavior if it attempts to "
                        "access the input tensor."
                    )

                def forward_pre_hook(module, *_) -> None:
                    for submodule, hook, with_kwargs in forward_pre_hooks:
                        if with_kwargs:
                            ret = hook(submodule, (), {})
                        else:
                            ret = hook(submodule, ())
                        if ret is not None:
                            raise RuntimeError(
                                "TEFusedResidualRMSNorm module does not expose "
                                "intermediate tensors, but submodule has "
                                "pre-forward hook that modifies input tensor."
                            )

                fused_impl.register_forward_pre_hook(forward_pre_hook)

            # Post-forward hooks
            if forward_post_hooks:
                warnings.warn(
                    "TEFusedResidualRMSNorm module has a submodule with a post-forward hook. "
                    "TEFusedResidualRMSNorm module does not expose intermediate tensors, "
                    "so the hook may have incorrect behavior if it attempts to "
                    "access the input or output tensors."
                )

                def forward_post_hook(module, *_) -> None:
                    for submodule, hook, with_kwargs in forward_post_hooks:
                        if with_kwargs:
                            ret = hook(submodule, (), {}, None)
                        else:
                            ret = hook(submodule, (), None)
                        if ret is not None:
                            raise RuntimeError(
                                "TEFusedResidualRMSNorm module does not expose "
                                "intermediate tensors, but submodule has "
                                "post-forward hook that modifies output tensor."
                            )

                fused_impl.register_forward_hook(forward_post_hook)

            # Backward hooks
            if backward_pre_hooks:
                raise RuntimeError(
                    "TEFusedResidualRMSNorm module does not support "
                    "submodules with pre-backward hooks"
                )
            if backward_post_hooks:
                raise RuntimeError(
                    "TEFusedResidualRMSNorm module does not support "
                    "submodules with post-backward hooks"
                )

        def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass with fused residual output.

            Args:
                hidden_states: Input tensor [s, b, h]

            Returns:
                Tuple of (normalized_output, residual), both [s, b, h]

            Note:
                Sequential.forward() automatically returns (output, extra_outputs...)
                when MakeExtraOutput is present, so we don't need manual unpacking.
            """

            # Construct fused impl lazily on first forward
            # (in case parameters are modified after __init__)
            if self._fused_impl is None:
                self._fused_impl = (self._make_fused_impl(),)

            # Apply fused implementation
            # Sequential returns (normalized_output, residual) automatically
            return self._fused_impl[0](hidden_states)

else:
    TEFusedResidualRMSNorm = None  # type: ignore[assignment, misc]


class TENorm:
    """A conditional wrapper to initialize an instance of
    Transformer-Engine's `LayerNorm` or `RMSNorm` based on input.

    Residual fusion is a two-level opt-in mechanism:

    1. Global capability: config.fused_residual_rmsnorm must be True (enables the feature)
    2. Local intent: has_residual=True must be passed at build site (declares this specific
       norm is followed by a residual connection)

    Fusion only happens when BOTH conditions are met.

    """

    # TODO should we ditch normalization config and just use spec to choose LayerNorm vs RMSNorm?
    def __new__(
        cls,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        has_residual: bool = False,
    ):
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        use_fused_residual = config.fused_residual_rmsnorm and has_residual
        if use_fused_residual and config.normalization != "RMSNorm":
            raise ValueError("Fused residual is only supported " "for RMSNorm normalization")

        if config.normalization == "LayerNorm":
            norm_module = te.pytorch.LayerNorm
        elif config.normalization == "RMSNorm":
            assert hasattr(
                te.pytorch, "RMSNorm"
            ), "Transformer-Engine >= v0.11 required to use this feature"
            if use_fused_residual:
                assert (
                    TEFusedResidualRMSNorm is not None
                ), "TEFusedResidualRMSNorm requires Transformer-Engine >= v1.13.0"
                norm_module = TEFusedResidualRMSNorm
            else:
                norm_module = te.pytorch.RMSNorm
        else:
            raise Exception("Only LayerNorm and RMSNorm are currently supported")

        instance = norm_module(
            normalized_shape=hidden_size,
            eps=eps,
            sequence_parallel=config.sequence_parallel,
            zero_centered_gamma=config.layernorm_zero_centered_gamma,
            **_get_extra_te_kwargs(config),
        )

        return cast(LayerNormInterface, instance)


class TELinear(te.pytorch.Linear):
    """Wrapper for the Transformer-Engine's `Linear` layer.

    Note that if Megatron's parallel_state has not been initialized
    yet, the tp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group().

    parallel_mode currently supports 3 different values:
        - "column": Split the weight matrix along output dimension (used in TEColumnParallelLinear)
        - "row": Split the weight matrix along input dimension (used in TERowParallelLinear)
        - "duplicated": No tensor parallelism and weight is duplicated across TP ranks
        - Note: For expert linear layers, we will disable communication logic here
                as TP communication is handled in token_dispatcher.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: Optional[str],
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        skip_weight_param_allocation: bool,
        tp_comm_buffer_name: Optional[str] = None,
        is_expert: bool = False,
        symmetric_ar_type: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        self.config = config

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache
        self.symmetric_ar_type = symmetric_ar_type
        if skip_weight_param_allocation:
            raise ValueError(
                "Transformer Engine linear layers do not support skip_weight_param_allocation"
            )

        extra_kwargs = _get_extra_te_kwargs(config)

        if self.config.delay_wgrad_compute:
            if is_te_min_version("2.3.0"):
                extra_kwargs["delay_wgrad_compute"] = self.config.delay_wgrad_compute
            else:
                raise RuntimeError("Only TE with version >=2.3.0 supports delay_wgrad_compute now.")

        if (
            self.config.tp_comm_overlap
            and tp_comm_buffer_name
            and tp_comm_buffer_name not in ["qkv", "proj", "fc1", "fc2"]
        ):
            self.config.tp_comm_overlap = False
            warnings.warn(
                f"The user buffer name {tp_comm_buffer_name} is not supported in"
                "Transformer Engine. Disabling TP communication overlap "
                "for this layer."
            )

        if is_te_min_version("0.8.0"):
            if self.config.tp_comm_overlap and parallel_mode != "duplicated":
                if is_te_min_version("1.5.0"):
                    # Use old overlap flags if they were supplied instead
                    extra_kwargs["ub_overlap_ag"] = (
                        self.config.tp_comm_overlap_ag
                        if hasattr(self.config, "tp_comm_overlap_ag")
                        else self.config.tp_comm_split_ag or self.config.tp_comm_atomic_ag
                    )
                    extra_kwargs["ub_overlap_rs"] = (
                        self.config.tp_comm_overlap_rs
                        if hasattr(self.config, "tp_comm_overlap_rs")
                        else self.config.tp_comm_split_rs or self.config.tp_comm_atomic_rs
                    )
                    # Disable ub overlap for experts.
                    if is_expert:
                        extra_kwargs["ub_overlap_ag"] = False
                        extra_kwargs["ub_overlap_rs"] = False
                else:
                    extra_kwargs["ub_split_ag"] = self.config.tp_comm_split_ag
                    extra_kwargs["ub_atomic_gemm_ag"] = self.config.tp_comm_atomic_ag
                    extra_kwargs["ub_split_rs"] = self.config.tp_comm_split_rs
                    extra_kwargs["ub_atomic_gemm_rs"] = self.config.tp_comm_atomic_rs
                    # Disable ub overlap for experts.
                    if is_expert:
                        extra_kwargs["ub_split_ag"] = False
                        extra_kwargs["ub_atomic_gemm_ag"] = False
                        extra_kwargs["ub_split_rs"] = False
                        extra_kwargs["ub_atomic_gemm_rs"] = False
                if is_te_min_version("1.0.0", check_equality=False):
                    assert (
                        tp_comm_buffer_name is not None
                    ), "Buffer name should be set to configure communication overlap settings"
                    extra_kwargs["ub_name"] = tp_comm_buffer_name

        if symmetric_ar_type is not None:
            assert is_torch_min_version("2.7.0a0"), "Must have at least torch version 2.7 or higher"
            assert is_te_min_version("2.3.0") or get_te_version() == PkgVersion(
                "2.3.0.dev0+39c0e70"
            ), "Must have at least TE version 2.3 or higher to use symmetric memory all reduce"
            extra_kwargs["symmetric_ar_type"] = symmetric_ar_type
        if parallel_mode == "duplicated":
            assert tp_group is None, "duplicated linear should not have tp_group set"
            tp_size = 1
        else:
            tp_size = get_pg_size(tp_group)

        self.expert_parallel = self.config.expert_model_parallel_size > 1
        if is_expert:
            rng_tracker_name = get_expert_parallel_rng_tracker_name()
        else:
            if parallel_mode == "duplicated":
                rng_tracker_name = get_data_parallel_rng_tracker_name()
            else:
                rng_tracker_name = None
        if is_te_min_version("1.7.0"):
            extra_kwargs["rng_tracker_name"] = rng_tracker_name

        te_parallel_mode = parallel_mode
        tp_group_for_te = tp_group
        if parallel_mode == "duplicated":
            # Handle non-parallel case
            tp_group_for_te = None
            tp_size = 1
            explicit_expert_comm = False
            te_parallel_mode = None
        else:
            # Disable communications in TE when using TP or EP by
            explicit_expert_comm = is_expert and (tp_size > 1 or self.expert_parallel)

            if explicit_expert_comm:
                if parallel_mode == "column":
                    output_size = divide(output_size, tp_size)
                elif parallel_mode == "row":
                    input_size = divide(input_size, tp_size)
                te_parallel_mode = None
                tp_size = 1
                tp_group_for_te = None

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            # Pass None if not initialized for backward compatibility with the ckpt converter.
            tp_group=tp_group_for_te if torch.distributed.is_initialized() else None,
            tp_size=tp_size,
            get_rng_state_tracker=(
                get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
            ),
            init_method=condition_init_method(config, init_method),
            bias=bias,
            return_bias=self.te_return_bias,
            parallel_mode=te_parallel_mode,
            **extra_kwargs,
        )
        self.te_quant_params: Optional[TEQuantizationParams] = None

        for param in self.parameters():
            if is_expert:
                # Reduce the gradient on the expert_data_parallel group for expert linear layers
                setattr(param, "allreduce", not self.expert_parallel)
            else:
                # Reduce the gradient on DP group
                setattr(param, "allreduce", True)
                if parallel_mode == "duplicated":
                    # Reduce the gradient further on the TP group since the weight is
                    # duplicated across TP ranks
                    setattr(param, "sequence_parallel", self.config.sequence_parallel)
                    # Mark as NOT tensor parallel since weight is duplicated
                    setattr(param, "tensor_model_parallel", False)

        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        self._tp_group = tp_group

    def finish_init(self, quantization_config: QuantizationConfig):
        """Post-init of quantization override"""
        if quantization_config is None:
            self.te_quant_params = None
        else:
            self.te_quant_params = TEQuantizationParams.parse_from_config(quantization_config)

    def will_execute_quantized(self, is_context_quantized: bool) -> bool:
        """Returns whether the module is configured to execute quantized."""
        return _get_should_context_be_quantized_params(
            self.te_quant_params, self.training, is_context_quantized
        )

    def forward(self, x):
        """Forward."""
        _is_first_microbatch = (
            None if self.disable_parameter_transpose_cache else self.is_first_microbatch
        )
        quant_context = _get_fp8_autocast_for_quant_params(self.te_quant_params, self.training)

        with quant_context:
            out = super().forward(x, is_first_microbatch=_is_first_microbatch)
        self.is_first_microbatch = False

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.te_return_bias:
            return out
        return out, None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Replicate cross TP/DP."""

        # Provide the dist-ckpt support when TELinear is directly used
        # It can only happen with duplicated parallel mode
        assert (
            self.parallel_mode is None
        ), "TELinear sharded_state_dict can only be used with duplicated parallel mode"
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            None,
            sharded_offsets,
            tp_group=self._tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )

    def backward_dw(self):
        """Compute weight gradients during the backward pass if delay_wgrad_compute is enabled."""
        if self.config.delay_wgrad_compute:
            super().backward_dw()


class TELayerNormColumnParallelLinear(te.pytorch.LayerNormLinear):
    """Wrapper for the Transformer-Engine's `LayerNormLinear` layer
    that combines layernorm and linear layers."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        stride: int = 1,
    ):
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        self.config = config

        if gather_output:
            raise ValueError("Transformer Engine linear layers do not support gather_output = True")

        if is_expert:
            raise ValueError("Transformer Engine linear layers do not yet support MoE")

        if skip_weight_param_allocation:
            raise ValueError(
                "Transformer Engine linear layers do not support skip_weight_param_allocation"
            )

        # TODO: For backward compatibility, remove in v0.15.
        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        self._tp_group = tp_group

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache
        extra_kwargs = _get_extra_te_kwargs(config)
        self.tp_size = get_pg_size(tp_group)
        self.tp_rank = get_pg_rank(tp_group)

        if self.config.delay_wgrad_compute:
            if is_te_min_version("2.3.0"):
                extra_kwargs["delay_wgrad_compute"] = self.config.delay_wgrad_compute
            else:
                raise RuntimeError("Only TE with version >=2.3.0 supports delay_wgrad_compute now.")

        # Only Transformer-Engine version >= 0.11.0 supports `RMSNorm`
        if is_te_min_version("0.11.0"):
            extra_kwargs["normalization"] = self.config.normalization
        elif self.config.normalization != "LayerNorm":
            te_version = get_te_version()
            raise ValueError(
                f"Transformer Engine v{te_version} does not support {self.config.normalization}."
            )

        if is_te_min_version("0.8.0"):
            if self.config.tp_comm_overlap:
                extra_kwargs["ub_bulk_wgrad"] = self.config.tp_comm_bulk_wgrad
                extra_kwargs["ub_bulk_dgrad"] = self.config.tp_comm_bulk_dgrad
                if is_te_min_version("1.5.0", check_equality=False):
                    # Use old overlap flags if they were supplied instead
                    extra_kwargs["ub_overlap_ag"] = (
                        self.config.tp_comm_overlap_ag
                        if hasattr(self.config, "tp_comm_overlap_ag")
                        else self.config.tp_comm_split_ag or self.config.tp_comm_atomic_ag
                    )
                    if is_te_min_version("1.6.0.dev0", check_equality=False):
                        extra_kwargs["ub_overlap_rs_dgrad"] = (
                            self.config.tp_comm_overlap_rs_dgrad
                            if hasattr(self.config, "tp_comm_overlap_rs_dgrad")
                            else False
                        )
                    if tp_comm_buffer_name == "qkv" and self.config.tp_comm_overlap_disable_qkv:
                        extra_kwargs["ub_overlap_ag"] = False
                        extra_kwargs["ub_overlap_rs_dgrad"] = False

                    if tp_comm_buffer_name == "fc1" and self.config.tp_comm_overlap_disable_fc1:
                        extra_kwargs["ub_overlap_ag"] = False
                        extra_kwargs["ub_overlap_rs_dgrad"] = False
                else:
                    extra_kwargs["ub_atomic_gemm_ag"] = self.config.tp_comm_atomic_ag
                    extra_kwargs["ub_split_ag"] = self.config.tp_comm_split_ag
                if is_te_min_version("1.0.0", check_equality=False):
                    assert (
                        tp_comm_buffer_name is not None
                    ), "Buffer name should be set to configure communication overlap settings"
                    extra_kwargs["ub_name"] = tp_comm_buffer_name

        if self.config.symmetric_ar_type is not None:
            assert is_torch_min_version("2.7.0a0"), "Must have at least torch version 2.7 or higher"
            assert is_te_min_version("2.3.0") or get_te_version() == PkgVersion(
                "2.3.0.dev0+39c0e70"
            ), "Must have at least TE version 2.3 or higher to use symmetric memory all reduce"
            extra_kwargs["symmetric_ar_type"] = self.config.symmetric_ar_type

        self.stride = stride

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            eps=self.config.layernorm_epsilon,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            tp_group=tp_group if torch.distributed.is_initialized() else None,
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=(
                get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
            ),
            init_method=(
                condition_init_method(config, init_method)
                if not config.use_cpu_initialization
                else lambda w: None
            ),
            bias=bias,
            return_bias=self.te_return_bias,
            parallel_mode="column",
            return_layernorm_output=False,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
            **extra_kwargs,
        )
        self.te_quant_params: Optional[TEQuantizationParams] = None

        # Set proper partition_stride
        setattr(self.weight, 'partition_stride', stride)
        if bias and hasattr(self, 'bias') and self.bias is not None:
            setattr(self.bias, 'partition_stride', stride)

        if config.use_cpu_initialization:
            output_size_per_partition = divide(output_size, self.tp_size)
            _ = _initialize_affine_weight_cpu(
                self.weight,
                output_size,
                input_size,
                output_size_per_partition,
                0,
                init_method=condition_init_method(config, init_method),
                stride=stride,
                return_master_weight=False,
                rank=self.tp_rank,
                world_size=self.tp_size,
                skip_set_tensor_parallel_attributes=True,
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(output_size_per_partition, dtype=config.params_dtype)
                )
                set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
                with torch.no_grad():
                    self.bias.zero_()
                setattr(self.bias, "allreduce", True)

    def finish_init(self, quantization_config: QuantizationConfig):
        """Post-init of quantization override"""
        if quantization_config is None:
            self.te_quant_params = None
        else:
            self.te_quant_params = TEQuantizationParams.parse_from_config(quantization_config)

    def will_execute_quantized(self, is_context_quantized: bool) -> bool:
        """Returns whether the module is configured to execute quantized."""
        return _get_should_context_be_quantized_params(
            self.te_quant_params, self.training, is_context_quantized
        )

    def forward(self, x):
        """Forward."""
        _is_first_microbatch = (
            None if self.disable_parameter_transpose_cache else self.is_first_microbatch
        )
        quant_context = _get_fp8_autocast_for_quant_params(self.te_quant_params, self.training)

        with quant_context:
            out = super().forward(x, is_first_microbatch=_is_first_microbatch)

        self.is_first_microbatch = False

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.te_return_bias:
            return out
        return out, None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {"weight": 0, "bias": 0},
            sharded_offsets,
            tp_group=self._tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )

    @override
    def extra_repr(self) -> str:
        """Extra context to add to the module's string representation."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.use_bias}, "
            f"TP={self.tp_size}"
        )

    def backward_dw(self):
        """Compute weight gradients during the backward pass if delay_wgrad_compute is enabled."""
        if self.config.delay_wgrad_compute:
            super().backward_dw()


class TEColumnParallelLinear(TELinear):
    """Wrapper for the Transformer-Engine's `Linear` layer
    but specialized similar to megatron's `ColumnParallelLinear` layer."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        stride: int = 1,
    ):
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        if gather_output:
            raise ValueError("Transformer Engine linear layers do not support gather_output = True")
        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        self._tp_group = tp_group
        world_size = get_pg_size(tp_group)
        rank = get_pg_rank(tp_group)
        self.stride = stride

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=(
                condition_init_method(config, init_method)
                if not config.use_cpu_initialization
                else lambda w: None
            ),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            symmetric_ar_type=config.symmetric_ar_type,
            tp_group=tp_group,
        )

        # Set proper partition_stride
        setattr(self.weight, 'partition_stride', stride)
        if bias and hasattr(self, 'bias') and self.bias is not None:
            setattr(self.bias, 'partition_stride', stride)

        if config.use_cpu_initialization:
            output_size_per_partition = divide(output_size, world_size)
            _ = _initialize_affine_weight_cpu(
                self.weight,
                output_size,
                input_size,
                output_size_per_partition,
                0,
                init_method=condition_init_method(config, init_method),
                stride=stride,
                return_master_weight=False,
                rank=rank,
                world_size=world_size,
                skip_set_tensor_parallel_attributes=True,
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(output_size_per_partition, dtype=config.params_dtype)
                )
                set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
                with torch.no_grad():
                    self.bias.zero_()
                setattr(self.bias, "allreduce", True)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {"weight": 0, "bias": 0},
            sharded_offsets,
            tp_group=self._tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )

    @override
    def extra_repr(self) -> str:
        """Extra context to add to the module's string representation."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.use_bias}, "
            f"TP={self.tp_size}"
        )

    def backward_dw(self):
        """Compute weight gradients during the backward pass if delay_wgrad_compute is enabled."""
        if self.config.delay_wgrad_compute:
            super().backward_dw()


class TERowParallelLinear(TELinear):
    """Wrapper for the Transformer-Engine's `Linear` layer
    but specialized similar to megatron's `RowParallelLinear` layer."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        if not input_is_parallel:
            raise ValueError(
                "Transformer Engine linear layers do not support input_is_parallel = False"
            )
        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        self._tp_group = tp_group

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=(
                condition_init_method(config, init_method)
                if not config.use_cpu_initialization
                else lambda w: None
            ),
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=False,
            # We don't currently use this for row parallel layers # pylint: disable=line-too-long
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            symmetric_ar_type=config.symmetric_ar_type,
            tp_group=tp_group,
        )
        if config.use_cpu_initialization:
            world_size = get_pg_size(tp_group)
            rank = get_pg_rank(tp_group)
            input_size_per_partition = divide(input_size, world_size)
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight,
                output_size,
                input_size,
                input_size_per_partition,
                1,
                init_method=condition_init_method(config, init_method),
                stride=1,
                return_master_weight=False,
                params_dtype=config.params_dtype,
                rank=rank,
                world_size=world_size,
                skip_set_tensor_parallel_attributes=True,
            )
            if bias:
                self.bias = Parameter(torch.empty(output_size, dtype=config.params_dtype))
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias.zero_()
                setattr(self.bias, "allreduce", True)
                setattr(self.bias, "sequence_parallel", config.sequence_parallel)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 1, bias not sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {"weight": 1},
            sharded_offsets,
            tp_group=self._tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )

    @override
    def extra_repr(self) -> str:
        """Extra context to add to the module's string representation."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.use_bias}, "
            f"TP={self.tp_size}"
        )

    def backward_dw(self):
        """Compute weight gradients during the backward pass if delay_wgrad_compute is enabled."""
        if self.config.delay_wgrad_compute:
            super().backward_dw()


class TEDotProductAttention(te.pytorch.DotProductAttention):
    """Wrapper for the Transformer-Engine's `DotProductAttention` layer
    that also has "flash attention" enabled.

    Note that if Megatron's parallel_state has not been initialized yet, the
    tp_group and cp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group() and set_context_parallel_group().
    """

    cp_stream: torch.cuda.Stream = None

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        num_splits: Optional[int] = None,
        cp_comm_type: Optional[str] = "p2p",
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        self.config = config
        self.te_forward_mask_type = False
        self.qkv_format: str = "sbhd"
        # Default to 1 split when batch-invariant mode is enabled, unless explicitly overridden
        self.num_splits: Optional[int] = (
            1 if (num_splits is None and self.config.batch_invariant_mode) else num_splits
        )

        if self.config.apply_query_key_layer_scaling != bool(
            int(os.getenv("NVTE_APPLY_QK_LAYER_SCALING", "0"))
        ):
            raise ValueError(
                f"apply_query_key_layer_scaling is {self.config.apply_query_key_layer_scaling} "
                f"but environment variable NVTE_APPLY_QK_LAYER_SCALING is "
                f"{os.getenv('NVTE_APPLY_QK_LAYER_SCALING')}. Transformer Engine does not support "
                f"setting query key layer scaling via argument, so these two must match."
            )

        extra_kwargs: dict[str, Any] = {}
        if is_te_min_version("0.11.0"):
            extra_kwargs["num_gqa_groups"] = self.config.num_query_groups
        elif self.config.num_query_groups != self.config.num_attention_heads:
            raise ValueError(
                f"Transformer Engine v{get_te_version()} does not support Grouped Query Attention, "
                f"use a newer version of Transformer Engine. "
                f"(num_query_groups ({self.config.num_query_groups}) != "
                f"num_attention_heads ({self.config.num_attention_heads}))"
            )

        if pg_collection is None:
            pg_collection = ProcessGroupCollection(
                tp=get_tensor_model_parallel_group(check_initialized=False),
                cp=get_context_parallel_group(check_initialized=False),
                hcp=get_hierarchical_context_parallel_groups(check_initialized=False),
            )
        else:
            assert hasattr(
                pg_collection, "tp"
            ), "TEDotProductAttention pg_collection must have tp pg"
            assert hasattr(
                pg_collection, "cp"
            ), "TEDotProductAttention pg_collection must have cp pg"
            if cp_comm_type == "a2a+p2p":
                assert hasattr(
                    pg_collection, "hcp"
                ), "TEDotProductAttention pg_collection must have hierarchical cp pg"
        self._tp_group = pg_collection.tp

        if is_te_min_version("0.10.0"):
            extra_kwargs["attention_type"] = attention_type
            # older version don't need attention_type

        if is_te_min_version("0.12.0", check_equality=False):
            self.te_forward_mask_type = True

        # This check is important as CP config can be disabled while having a valid CP group
        # Example - Disabling CP for encoder while a valid CP group exists for decoder
        if self.config.context_parallel_size > 1:
            assert is_te_min_version(
                "1.0.0"
            ), "Only Transformer-Engine version >= 1.0.0 supports context parallelism!"
            if getattr(TEDotProductAttention, "cp_stream") is None:
                TEDotProductAttention.cp_stream = torch.cuda.Stream()
            extra_kwargs["cp_group"] = pg_collection.cp
            extra_kwargs["cp_global_ranks"] = torch.distributed.get_process_group_ranks(
                pg_collection.cp
            )
            extra_kwargs["cp_stream"] = TEDotProductAttention.cp_stream
            if is_te_min_version("1.10.0"):
                if cp_comm_type is None:
                    extra_kwargs["cp_comm_type"] = "p2p"
                elif cp_comm_type == "a2a+p2p":
                    assert is_te_min_version("1.12.0"), (
                        f"Transformer-Engine v{get_te_version()} must be >= 1.12.0 to support"
                        "hierarchical cp commucation."
                    )
                    extra_kwargs["cp_comm_type"] = "a2a+p2p"
                    extra_kwargs["cp_group"] = get_hierarchical_context_parallel_groups(
                        check_initialized=False
                    )
                else:
                    extra_kwargs["cp_comm_type"] = cp_comm_type

        if self.config.deterministic_mode:
            if int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")) != 0:
                raise RuntimeError(
                    "deterministic_mode is on and we are using DotProductAttention from "
                    "Transformer Engine, but NVTE_ALLOW_NONDETERMINISTIC_ALGO is not 0. "
                    f"Currently set to: {os.getenv('NVTE_ALLOW_NONDETERMINISTIC_ALGO', 'not set')}."
                )

        if is_layer_window_attention(
            config.window_size, config.window_attn_skip_freq, layer_number
        ):
            # Check version
            assert is_te_min_version("1.2.0"), (
                f"Transformer-Engine v{get_te_version()} must be >= 1.2.0 to support"
                "sliding window attention."
            )
            extra_kwargs["window_size"] = config.window_size

        if is_te_min_version("1.10.0"):
            # TE 1.10.0 introduces the ability to set the different k and v channels
            kv_channels = (
                (k_channels, v_channels)
                if k_channels is not None and v_channels is not None
                else self.config.kv_channels
            )
            extra_kwargs["softmax_scale"] = softmax_scale
        else:
            kv_channels = self.config.kv_channels

        if self.config.softmax_type != "vanilla":
            assert is_te_min_version("2.8.0"), (
                f"Transformer-Engine v{get_te_version()} must be >= 2.8.0 to support"
                "`softmax_type`."
            )
            extra_kwargs["softmax_type"] = self.config.softmax_type

        self.kept_packed_seq_params = set(
            field.name for field in dataclasses.fields(PackedSeqParams)
        )

        if get_te_version() < PkgVersion("1.3.0"):
            # TE 1.3.0 introduces precomputing max_seqlen to remove unnecessary kernels and D2H
            # copies (#555)
            # These two arguments did not exist prior to 1.3.0
            self.kept_packed_seq_params.discard("max_seqlen_q")
            self.kept_packed_seq_params.discard("max_seqlen_kv")

        if get_te_version() < PkgVersion("1.10.0"):
            # TE 1.8.0 introduces cu_seqlens_padded which is the cu_seqlens with paddings counted
            # in each individual sequence in THD format dataset
            # These two arguments did not exist prior to 1.8.0. Full support added in 1.10.0 (#1012)
            self.kept_packed_seq_params.discard("cu_seqlens_q_padded")
            self.kept_packed_seq_params.discard("cu_seqlens_kv_padded")

        # total_tokens and seq_idx are only for Mamba and should not be forwarded to TE attention.
        self.kept_packed_seq_params.discard("total_tokens")
        self.kept_packed_seq_params.discard("seq_idx")

        if config.qk_clip or config.log_max_attention_logit:
            # qk-clip is only supported in TE 2.9.0 and later
            assert is_te_min_version("2.9.0"), "qk-clip is only supported in TE 2.9.0 and later"

            # TE 2.9.0 introduces return_max_logit for qk-clip getting the max attention logits
            extra_kwargs["return_max_logit"] = True
            self.current_max_attn_logits = None

        super().__init__(
            num_attention_heads=self.config.num_attention_heads,
            kv_channels=kv_channels,
            attention_dropout=(
                self.config.attention_dropout if attention_dropout is None else attention_dropout
            ),
            attn_mask_type=attn_mask_type.name,
            sequence_parallel=self.config.sequence_parallel,
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=(
                get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
            ),
            tp_group=pg_collection.tp,
            layer_number=layer_number,
            **extra_kwargs,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        attn_mask_type: AttnMaskType,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        num_splits: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward."""
        if packed_seq_params is not None:
            # If Dynamic CP group is provided, update TE DPA CP group
            if packed_seq_params.local_cp_size is not None:
                if packed_seq_params.local_cp_size == 1:
                    super().set_context_parallel_group(None, None, None, self.cp_comm_type)
                else:
                    assert (
                        packed_seq_params.cp_group is not None
                    ), "cp_group is not set in packed_seq_params for dynamic CP"
                    self.cp_group = packed_seq_params.cp_group
                    if TEDotProductAttention.cp_stream is None:
                        TEDotProductAttention.cp_stream = torch.cuda.Stream()
                    super().set_context_parallel_group(
                        self.cp_group,
                        torch.distributed.get_process_group_ranks(self.cp_group),
                        TEDotProductAttention.cp_stream,
                        self.cp_comm_type,
                    )
            self.kept_packed_seq_params.discard("cp_group")
            self.kept_packed_seq_params.discard("local_cp_size")

        # Default to constructor-provided num_splits unless explicitly overridden
        if num_splits is None:
            num_splits = self.num_splits
        if num_splits is not None:
            assert is_te_min_version("2.10.0"), (
                f"Transformer-Engine v{get_te_version()} must be >= 2.10.0 to support" "num_splits."
            )

        packed_seq_kwargs = (
            {key: getattr(packed_seq_params, key) for key in self.kept_packed_seq_params}
            if packed_seq_params is not None
            else {}
        )
        qkv_format = packed_seq_kwargs.get('qkv_format', self.qkv_format)

        attention_bias_kwargs = {}
        if attention_bias is not None:
            assert is_te_min_version("1.2.0"), (
                f"Transformer-Engine v{get_te_version()} must be >= 1.2.0 to support"
                "`attention_bias`."
            )
            attention_bias_kwargs = dict(
                core_attention_bias_type="post_scale_bias", core_attention_bias=attention_bias
            )

        if attn_mask_type == AttnMaskType.no_mask and self.config.window_size is not None:
            if (qkv_format == "bshd" and query.size(1) == 1) or (
                qkv_format == "sbhd" and query.size(0) == 1
            ):
                #  need to change mask type for SWA inference decode stage.
                attn_mask_type = AttnMaskType.causal_bottom_right
        if self.te_forward_mask_type:
            if qkv_format == "thd" and is_te_min_version("1.7.0"):
                # thd format uses flash attention with cuDNN kernel which requires is_padding=True,
                # so the only acceptable mask types are `padding_causal` and `padding`. These do not
                # necessarily indicate there are padded tokens in the sequence.
                if attn_mask_type == AttnMaskType.causal:
                    attn_mask_type = AttnMaskType.padding_causal
                elif attn_mask_type == AttnMaskType.no_mask:
                    attn_mask_type = AttnMaskType.padding
            _fa_kwargs = dict(
                attn_mask_type=attn_mask_type.name, **attention_bias_kwargs, **packed_seq_kwargs
            )
            if num_splits is not None:
                _fa_kwargs["num_splits"] = num_splits

            core_attn_out = super().forward(query, key, value, attention_mask, **_fa_kwargs)

            if self.config.qk_clip or self.config.log_max_attention_logit:
                # qk-clip is only supported in TE 2.9.0 and later
                assert is_te_min_version("2.9.0"), "qk-clip is only supported in TE 2.9.0 and later"

                # Update Q K outside of TE Attention API
                core_attn_out, batch_max_attention_logits = core_attn_out

                # Update QK_Clip balancing eta
                if self.current_max_attn_logits is None:
                    self.current_max_attn_logits = batch_max_attention_logits
                else:
                    self.current_max_attn_logits = torch.max(
                        self.current_max_attn_logits, batch_max_attention_logits
                    )

        else:
            _fa_kwargs = dict(**attention_bias_kwargs, **packed_seq_kwargs)
            if num_splits is not None:
                _fa_kwargs["num_splits"] = num_splits
            core_attn_out = super().forward(query, key, value, attention_mask, **_fa_kwargs)

        return core_attn_out

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Sharded state dict for the learnable softmax offset parameter"""
        if self.config.softmax_type == "learnable":
            state_dict = self.state_dict(prefix="", keep_vars=True)
        else:
            state_dict = {}
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {'softmax_offset': 0},
            sharded_offsets,
            tp_group=self._tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )


if HAVE_TE and is_te_min_version("1.9.0.dev0"):

    class TEGroupedLinear(te.pytorch.GroupedLinear):
        """
        Wrapper for the Transformer-Engine's `GroupedLinear` layer.

        Note that if Megatron's parallel_state has not been initialized
        yet, the tp_group passed to TE will be None and must be set later
        via set_tensor_parallel_group().
        """

        def __init__(
            self,
            num_gemms: int,
            input_size: int,
            output_size: int,
            *,
            parallel_mode: Optional[str],
            config: ModelParallelConfig,
            init_method: Callable,
            bias: bool,
            skip_bias_add: bool,
            is_expert: bool = False,
            tp_comm_buffer_name: Optional[str] = None,
            pg_collection: Optional[ProcessGroupCollection] = None,
        ):
            self.config = config

            # TE returns a zero length Tensor when bias=False and
            # return_bias=True, but we prefer None.  So in that case we
            # tell TE to not return the bias, and return None
            # ourselves. This way our forward always returns two values
            # and we don't have to deal with the zero length Tensor.
            self.te_return_bias = skip_bias_add and bias
            self.is_first_microbatch = True
            self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache

            extra_kwargs = _get_extra_te_kwargs(config)
            self.delay_wgrad_compute = (
                self.config.delay_wgrad_compute
                or self.config.overlap_dispatch_backward_with_experts_wgrad
            )

            if self.delay_wgrad_compute:
                if is_te_min_version("2.3.0"):
                    extra_kwargs["delay_wgrad_compute"] = True
                else:
                    raise RuntimeError(
                        "Only TE with version >=2.3.0 supports delay_wgrad_compute now."
                    )

            extra_kwargs["ub_name"] = tp_comm_buffer_name

            self.expert_parallel = self.config.expert_model_parallel_size > 1
            if is_expert:
                extra_kwargs["rng_tracker_name"] = get_expert_parallel_rng_tracker_name()

            # The comms between TP and EP group is explicitly handled by MoE token dispatcher.
            # So we disable comms by making TE agnostic of model parallel.
            if pg_collection is None:
                pg_collection = ProcessGroupCollection.use_mpu_process_groups()
            self._pg_collection = pg_collection
            assert is_expert, "TEGroupedLinear only supports expert parallelism"
            tp_group = pg_collection.expt_tp
            self._tp_group = tp_group
            tp_size = get_pg_size(tp_group)
            tp_group_for_te = tp_group

            self.explicit_expert_comm = is_expert and (tp_size > 1 or self.expert_parallel)

            # Save original parallel_mode before clearing it for explicit_expert_comm.
            # When explicit_expert_comm is True, Megatron handles TP communication externally
            # and passes parallel_mode=None to TE. This causes TE to set partition_dim=0 on
            # all weights (its default for non-parallel mode). We need to fix this after init
            # so that refit/resharding can correctly identify which dimension is TP-partitioned.
            original_parallel_mode = parallel_mode

            if self.explicit_expert_comm:
                if parallel_mode == "column":
                    output_size = divide(output_size, tp_size)
                elif parallel_mode == "row":
                    input_size = divide(input_size, tp_size)
                parallel_mode = None
                tp_size = 1
                tp_group_for_te = None

            if is_te_min_version("2.14.0"):
                extra_kwargs["single_grouped_weight"] = getattr(
                    config, "moe_single_grouped_weight", False
                )
                extra_kwargs["single_grouped_bias"] = getattr(
                    config, "moe_single_grouped_bias", False
                )

            super().__init__(
                num_gemms=num_gemms,
                in_features=input_size,
                out_features=output_size,
                sequence_parallel=self.config.sequence_parallel,
                fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
                tp_group=tp_group_for_te if torch.distributed.is_initialized() else None,
                tp_size=tp_size,
                get_rng_state_tracker=(
                    get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
                ),
                init_method=condition_init_method(config, init_method),
                bias=bias,
                return_bias=self.te_return_bias,
                parallel_mode=parallel_mode,
                **extra_kwargs,
            )
            self.te_quant_params: Optional[TEQuantizationParams] = None
            for param in self.parameters():
                setattr(param, "allreduce", not (is_expert and self.expert_parallel))

            def normalize_grouped_parameter_keys(
                self,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            ):
                """Make grouped checkpoint keys compatible across parameter layouts."""

                def maybe_remap_param(param_name: str, single_grouped: bool) -> None:
                    grouped_key = f"{prefix}{param_name}"
                    indexed_keys = [
                        f"{prefix}{param_name}{gemm_idx}" for gemm_idx in range(self.num_gemms)
                    ]
                    has_grouped_key = grouped_key in state_dict
                    has_any_indexed_key = any(key in state_dict for key in indexed_keys)
                    has_all_indexed_keys = all(key in state_dict for key in indexed_keys)

                    if single_grouped:
                        if has_grouped_key or not has_all_indexed_keys:
                            return
                        state_dict[grouped_key] = torch.stack(
                            [state_dict.pop(key) for key in indexed_keys], dim=0
                        )
                    else:
                        if has_any_indexed_key or not has_grouped_key:
                            return
                        split_tensors = self._split_grouped_checkpoint_tensor(
                            state_dict.pop(grouped_key), grouped_key
                        )
                        for gemm_idx, tensor in enumerate(split_tensors):
                            state_dict[f"{prefix}{param_name}{gemm_idx}"] = tensor

                maybe_remap_param("weight", getattr(self, "single_grouped_weight", False))
                if self.use_bias:
                    maybe_remap_param("bias", getattr(self, "single_grouped_bias", False))

            self._register_load_state_dict_pre_hook(
                normalize_grouped_parameter_keys, with_module=True
            )

            # Explicitly stamp partition_dim and partition_stride on expert weight
            # tensors when explicit_expert_comm cleared parallel_mode.  TE ≤2.12
            # set these internally; TE ≥2.13 no longer does (parallel_mode=None
            # is passed due to explicit_expert_comm).  The resharding/refit planner
            # relies on partition_dim to correctly plan TP gather/scatter operations.
            # NOTE: we intentionally do NOT stamp tensor_model_parallel here —
            # doing so would change num-zeros gradient counting.
            if self.explicit_expert_comm and original_parallel_mode in ("column", "row"):
                part_dim = 0 if original_parallel_mode == "column" else 1
                for i in range(num_gemms):
                    weight = getattr(self, f"weight{i}", None)
                    if weight is not None:
                        setattr(weight, "partition_dim", part_dim)
                        setattr(weight, "partition_stride", 1)

            def merge_extra_states(
                self,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            ):
                """
                Merge multiple "_extra_state" into one.
                """
                self.init_fp8_metadata(num_gemms=self.num_gemms)
                # When resume training, loading ckpt is out of fp8_autocast context.
                # So we need to manually detect from the state_dict.
                fp8_checkpoint = any("_extra_state" in str(key) for key in state_dict.keys())

                if not fp8_checkpoint:
                    return

                try:
                    state_list = [
                        state_dict.pop(f"{prefix}_extra_state{i}") for i in range(1, self.num_gemms)
                    ]
                except KeyError:
                    # "_extra_state{i}" only exists for dist-ckpt. Return for torch native ckpt.
                    return

                # Early return conditions:
                # 1. Empty state_dict
                # 2. Empty state_list
                # 3. _extra_state is None
                # 4. _extra_state does not contain any information
                if (
                    not state_dict
                    or not state_list
                    or state_dict.get(f"{prefix}_extra_state") is None
                    or self._decode_extra_state(state_dict[f"{prefix}_extra_state"]) is None
                ):
                    return

                state_list = [state_dict.pop(f"{prefix}_extra_state")] + state_list
                state_list = [self._decode_extra_state(state) for state in state_list]
                extra_fp8_variables = state_list[0]["extra_fp8_variables"]
                extra_fp8_variables["num_gemms"] = self.num_gemms
                extra_state = {"extra_fp8_variables": extra_fp8_variables}
                # TE 2.0 adds recipe in extra_state
                if is_te_min_version("2.0.0"):
                    self.fp8_meta["recipe"] = state_list[0]["recipe"]
                    extra_state["recipe"] = self.fp8_meta["recipe"]
                # Only delayed scaling has global fp8 meta tensors. We're not using
                # self.fp8_meta["recipe"].delayed() because it's available in TE 2.0 and later.
                if isinstance(self.fp8_meta["recipe"], te.common.recipe.DelayedScaling):
                    extra_state.update(
                        {
                            "scale_fwd": torch.cat(
                                [state["scale_fwd"].view(-1, 1) for state in state_list], dim=1
                            ).view(-1),
                            "amax_history_fwd": torch.cat(
                                [state["amax_history_fwd"].view(-1, 1) for state in state_list],
                                dim=1,
                            ).view(self.fp8_meta["recipe"].amax_history_len, -1),
                            "scale_bwd": torch.cat(
                                [state["scale_bwd"].view(-1, 1) for state in state_list], dim=1
                            ).view(-1),
                            "amax_history_bwd": torch.cat(
                                [state["amax_history_bwd"].view(-1, 1) for state in state_list],
                                dim=1,
                            ).view(self.fp8_meta["recipe"].amax_history_len, -1),
                        }
                    )
                    # TE 2.0 removes scale_inv_fwd and scale_inv_bwd
                    if not is_te_min_version("2.0.0"):
                        extra_state.update(
                            {
                                "scale_inv_fwd": torch.cat(
                                    [state["scale_inv_fwd"].view(-1, 1) for state in state_list],
                                    dim=1,
                                ).view(-1),
                                "scale_inv_bwd": torch.cat(
                                    [state["scale_inv_bwd"].view(-1, 1) for state in state_list],
                                    dim=1,
                                ).view(-1),
                            }
                        )
                state_dict[f"{prefix}_extra_state"] = self._encode_extra_state(extra_state)

            self._register_load_state_dict_pre_hook(merge_extra_states, with_module=True)

        def _split_grouped_checkpoint_tensor(
            self, tensor: torch.Tensor, checkpoint_key: str
        ) -> list[torch.Tensor]:
            """Split grouped checkpoint tensor into one tensor per GEMM."""
            if hasattr(tensor, "split_into_quantized_tensors") and callable(
                tensor.split_into_quantized_tensors
            ):
                grouped_tensors = getattr(tensor, "quantized_tensors", None)
                if grouped_tensors is None:
                    grouped_tensors = tensor.split_into_quantized_tensors()
                if len(grouped_tensors) != self.num_gemms:
                    raise RuntimeError(
                        f"Grouped checkpoint tensor {checkpoint_key} has {len(grouped_tensors)} "
                        f"groups, expected {self.num_gemms}."
                    )
                return list(grouped_tensors)
            if tensor.ndim > 0 and tensor.shape[0] == self.num_gemms:
                return list(tensor.unbind(dim=0))
            if tensor.ndim > 0 and tensor.shape[0] % self.num_gemms == 0:
                return list(torch.chunk(tensor, self.num_gemms, dim=0))
            raise RuntimeError(
                f"Cannot split checkpoint tensor {checkpoint_key} with shape {tuple(tensor.shape)} "
                f"into {self.num_gemms} GEMM shards."
            )

        def finish_init(self, quantization_config: QuantizationConfig):
            """Post-init of quantization override"""
            if quantization_config is None:
                self.te_quant_params = None
            else:
                self.te_quant_params = TEQuantizationParams.parse_from_config(quantization_config)

        def will_execute_quantized(self, is_context_quantized: bool) -> bool:
            """Returns whether the module is configured to execute quantized."""
            return _get_should_context_be_quantized_params(
                self.te_quant_params, self.training, is_context_quantized
            )

        def forward(self, x, m_splits):
            """Forward."""
            _is_first_microbatch = (
                None if self.disable_parameter_transpose_cache else self.is_first_microbatch
            )
            quant_context = _get_fp8_autocast_for_quant_params(self.te_quant_params, self.training)

            with quant_context:
                out = super().forward(x, m_splits, is_first_microbatch=_is_first_microbatch)
            self.is_first_microbatch = False

            # TE only returns a tuple when return_bias is True, otherwise
            # it returns a single Tensor, we always want to return two
            # values regardless of the arguments.
            if self.te_return_bias:
                return out
            return out, None

        def _encode_extra_state(self, state):
            # TE 2.0 changed the format of extra_state to be a byte tensor
            if is_te_min_version("2.0.0"):
                torch.cuda.synchronize()
                state_serialized = bytearray(pickle.dumps(state))
                state_serialized = torch.frombuffer(state_serialized, dtype=torch.uint8)
            else:
                state_serialized = io.BytesIO()
                torch.save(state, state_serialized)
            return state_serialized

        def _decode_extra_state(self, state):
            if isinstance(state, torch.Tensor):
                # No FP8 is indicated by an empty tensor we don't need to unpickle.
                if state.numel() == 0:
                    return
                return pickle.loads(state.detach().cpu().numpy().tobytes())
            elif isinstance(state, io.BytesIO):
                state.seek(0)
                return torch.load(state, map_location="cuda")
            else:
                raise RuntimeError("Unsupported checkpoint format.")

        def _split_extra_state(self, state):
            fp8_checkpoint = self.fp8_meta["fp8_checkpoint"] or self.fp8 or self.fp8_calibration

            if not fp8_checkpoint:
                return [state] * self.num_gemms

            state = self._decode_extra_state(state)
            extra_states = []
            extra_fp8_variables = state["extra_fp8_variables"]
            extra_fp8_variables["num_gemms"] = 1
            for gemm_idx in range(self.num_gemms):
                tmp_state = {"extra_fp8_variables": extra_fp8_variables}
                # TE 2.0 adds recipe in extra_state
                if is_te_min_version("2.0.0"):
                    tmp_state["recipe"] = state["recipe"]
                # Only delayed scaling has global fp8 meta tensors. We're not using
                # self.fp8_meta["recipe"].delayed() because it's available in TE 2.0 and later.
                if isinstance(self.fp8_meta["recipe"], te.common.recipe.DelayedScaling):
                    tmp_state.update(
                        {
                            "scale_fwd": state["scale_fwd"].view(3, -1)[:, gemm_idx],
                            "amax_history_fwd": state["amax_history_fwd"].view(
                                self.fp8_meta["recipe"].amax_history_len, 3, -1
                            )[:, :, gemm_idx],
                            "scale_bwd": state["scale_bwd"].view(2, -1)[:, gemm_idx],
                            "amax_history_bwd": state["amax_history_bwd"].view(
                                self.fp8_meta["recipe"].amax_history_len, 2, -1
                            )[:, :, gemm_idx],
                        }
                    )
                    # TE 2.0 removes scale_inv_fwd and scale_inv_bwd
                    if not is_te_min_version("2.0.0"):
                        tmp_state.update(
                            {
                                "scale_inv_fwd": state["scale_inv_fwd"].view(3, -1)[:, gemm_idx],
                                "scale_inv_bwd": state["scale_inv_bwd"].view(2, -1)[:, gemm_idx],
                            }
                        )
                extra_states.append(self._encode_extra_state(tmp_state))
            return extra_states

        def _sharded_state_dict_grouped(
            self, tp_axis_map, prefix="", sharded_offsets=(), metadata=None
        ):
            """
            prefix should be module_name to make keys identical to sequetial ones.
            """
            singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
            sharded_state_dict = {}
            full_state_dict = self.state_dict(prefix="", keep_vars=True)
            grouped_split_cache = {}

            def get_gemm_tensor(param_name: str, gemm_idx: int) -> torch.Tensor:
                indexed_name = f"{param_name}{gemm_idx}"
                if indexed_name in full_state_dict:
                    return full_state_dict[indexed_name]
                if param_name not in full_state_dict:
                    raise KeyError(indexed_name)
                if param_name not in grouped_split_cache:
                    grouped_split_cache[param_name] = self._split_grouped_checkpoint_tensor(
                        full_state_dict[param_name], param_name
                    )
                grouped_splits = grouped_split_cache[param_name]
                return grouped_splits[gemm_idx]

            num_global_experts = get_pg_size(self._pg_collection.ep) * self.num_gemms
            local_expert_indices_offset = get_pg_rank(self._pg_collection.ep) * self.num_gemms
            ep_axis = len(sharded_offsets)
            extra_states = self._split_extra_state(full_state_dict["_extra_state"])
            for gemm_idx in range(self.num_gemms):
                global_expert_idx = local_expert_indices_offset + gemm_idx
                state_dict = {
                    f"{gemm_idx}.weight": get_gemm_tensor("weight", gemm_idx),
                    f"{gemm_idx}._extra_state": extra_states[gemm_idx],
                }
                if self.use_bias:
                    state_dict[f"{gemm_idx}.bias"] = get_gemm_tensor("bias", gemm_idx)
                if singleton_local_shards:
                    expert_prefix = f"{global_expert_idx}.{prefix}"
                    new_sharded_offsets = sharded_offsets
                else:
                    expert_prefix = prefix
                    new_sharded_offsets = (
                        *sharded_offsets,
                        (ep_axis, global_expert_idx, num_global_experts),
                    )
                sub_sd = make_sharded_tensors_for_checkpoint(
                    state_dict,
                    '',
                    tp_axis_map,
                    new_sharded_offsets,
                    tp_group=self._tp_group,
                    dp_cp_group=metadata["dp_cp_group"],
                )
                # Remove expert layers indexing from sharded keys
                replace_prefix_for_sharding(sub_sd, f"{gemm_idx}.", expert_prefix)
                sharded_state_dict.update(
                    {
                        f"{prefix}weight{gemm_idx}": sub_sd[f"{gemm_idx}.weight"],
                        f"{prefix}_extra_state{'' if gemm_idx == 0 else gemm_idx}": sub_sd[
                            f"{gemm_idx}._extra_state"
                        ],
                    }
                )
                if self.use_bias:
                    sharded_state_dict[f"{prefix}bias{gemm_idx}"] = sub_sd[f"{gemm_idx}.bias"]
            # Adjust replica ids - replication along DP modulo EP
            for k, sh_ten in sharded_state_dict.items():
                replica_id = sh_ten.replica_id
                assert (
                    len(replica_id) == 3
                ), f"Expected replica_id for {k} to be in (PP, TP, DP) format, got: {replica_id}"
                if getattr(sh_ten, "is_data_parallel_fully_shard", False):
                    edp_replica_id = 0
                else:
                    edp_replica_id = get_pg_rank(self._pg_collection.expt_dp)
                sh_ten.replica_id = (*replica_id[:2], edp_replica_id)
            return sharded_state_dict

        def backward_dw(self):
            """
            Compute weight gradients during the backward pass
            if delay_wgrad_compute is enabled.
            """
            if self.delay_wgrad_compute:
                super().backward_dw()

    class TEColumnParallelGroupedLinear(TEGroupedLinear):
        """
        Wrapper for the Transformer-Engine's `GroupedLinear` layer but specialized
        to column-parallel style.
        """

        def __init__(
            self,
            num_gemms: int,
            input_size: int,
            output_size: int,
            *,
            config: ModelParallelConfig,
            init_method: Callable,
            bias: bool,
            skip_bias_add: bool,
            is_expert: bool,
            tp_comm_buffer_name: Optional[str] = None,
            pg_collection: Optional[ProcessGroupCollection] = None,
        ):
            super().__init__(
                num_gemms=num_gemms,
                input_size=input_size,
                output_size=output_size,
                parallel_mode="column",
                config=config,
                init_method=condition_init_method(config, init_method),
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=tp_comm_buffer_name,
                pg_collection=pg_collection,
            )

        def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
            """
            For each gemm, sharding along axis 0, bias sharded.
            Assume sharded_offsets[-1] is the expert parallel offset.
            """
            tp_axis_map = {}
            for gemm_idx in range(self.num_gemms):
                tp_axis_map.update({f"{gemm_idx}.weight": 0, f"{gemm_idx}.bias": 0})
            return super()._sharded_state_dict_grouped(
                tp_axis_map, prefix, sharded_offsets, metadata
            )

    class TERowParallelGroupedLinear(TEGroupedLinear):
        """
        Wrapper for the Transformer-Engine's `GroupedLinear` layer but specialized
        to row-parallel style.
        """

        def __init__(
            self,
            num_gemms: int,
            input_size: int,
            output_size: int,
            *,
            config: ModelParallelConfig,
            init_method: Callable,
            bias: bool,
            skip_bias_add: bool,
            is_expert: bool,
            tp_comm_buffer_name: Optional[str] = None,
            pg_collection: Optional[ProcessGroupCollection] = None,
        ):
            super().__init__(
                num_gemms=num_gemms,
                input_size=input_size,
                output_size=output_size,
                parallel_mode="row",
                config=config,
                init_method=condition_init_method(config, init_method),
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=tp_comm_buffer_name,
                pg_collection=pg_collection,
            )

        def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
            """
            For each gemm, sharding along axis 1, bias not sharded.
            Assume sharded_offsets[-1] is the expert parallel offset.
            """
            tp_axis_map = {f"{gemm_idx}.weight": 1 for gemm_idx in range(self.num_gemms)}
            return super()._sharded_state_dict_grouped(
                tp_axis_map, prefix, sharded_offsets, metadata
            )

else:
    TEGroupedLinear = None  # type: ignore[assignment, misc]
    TEColumnParallelGroupedLinear = None  # type: ignore[assignment, misc]
    TERowParallelGroupedLinear = None  # type: ignore[assignment, misc]


if HAVE_TE and is_te_min_version("1.13.0"):

    class _MegatronQuantizedBasicLinear(te.pytorch.ops.BasicOperation):
        """BasicLinear-compatible op that quantizes weights before TE functional forward.

        This is a correctness/debug path for FP4: the registered parameter remains BF16, but the
        weight quantizer is invoked explicitly here instead of inside TE BasicLinear.
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            *,
            device: Optional[torch.device | str] = None,
            dtype: Optional[torch.dtype] = None,
            tensor_parallel_mode: Optional[str] = None,
            tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
            sequence_parallel: bool = False,
            rng_state_tracker_function: Optional[Callable[[], Any]] = None,
            accumulate_into_main_grad: bool = False,
            userbuffers_options: Optional[dict[str, Any]] = None,
        ) -> None:
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features
            dtype = torch.get_default_dtype() if dtype is None else dtype
            if dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise ValueError(f"Supported dtypes are float32, float16, bfloat16 (got {dtype})")

            (
                self.tensor_parallel_mode,
                self.tensor_parallel_group,
                self.tensor_parallel_size,
                self.sequence_parallel,
                self.local_in_features,
                self.local_out_features,
            ) = te.pytorch.ops.BasicLinear._canonicalize_tensor_parallelism(
                mode=tensor_parallel_mode,
                process_group=tensor_parallel_group,
                sequence_parallel=sequence_parallel,
                in_features=in_features,
                out_features=out_features,
            )

            weight = torch.empty(
                self.local_out_features,
                self.local_in_features,
                device=device,
                dtype=dtype,
            )
            self.weight: torch.nn.Parameter
            self.register_parameter("weight", torch.nn.Parameter(weight))
            self._rng_state_tracker_function = rng_state_tracker_function
            self._accumulate_into_main_grad = accumulate_into_main_grad
            self._userbuffers_options = userbuffers_options
            self._fp4_debug_weight_cache = None
            self._fp4_debug_weight_cache_compare_prints = 0
            self._fp4_debug_weight_cache_missing_prints = 0
            self._fp4_debug_weight_cache_skip_prints = 0
            self._fp4_debug_weight_cache_trace_prints = 0
            self._fp4_persistent_weight_shadow = None
            self._fp4_persistent_weight_shadow_source = None
            self._fp4_persistent_weight_shadow_compare_prints = 0
            self._fp4_debug_pending_compare_message = None
            self._fp4_debug_weight_cache_rowwise_usage = True
            self._fp4_debug_weight_cache_columnwise_usage = True
            self._fp4_debug_weight_cache_refresh_enabled = False
            self._fp4_debug_weight_cache_source = None
            self._fp4_debug_weight_compute_source = None
            self._register_fp4_weight_cache_refresh_hook()

        def _register_fp4_weight_cache_refresh_hook(self) -> None:
            self.weight._fp4_megatron_refresh_weight_cache_from_all_gather = (
                self._refresh_fp4_debug_weight_cache_from_all_gather
            )
            self.weight._fp4_megatron_skip_param_data_copy_after_all_gather = (
                self._skip_param_data_copy_after_all_gather
            )
            self.weight._fp4_megatron_unmap_param_data_from_ddp_buffer = (
                self._unmap_param_data_from_ddp_buffer
            )
            self.weight._fp4_megatron_drop_persistent_param_data = (
                self._drop_persistent_param_data
            )

        def num_quantizers(self, mode: str) -> int:
            if mode == "forward":
                return 2
            if mode == "backward":
                return 1
            return 0

        def pre_fuser_forward(self, *, requires_grad: bool) -> None:
            super().pre_fuser_forward(requires_grad=requires_grad)
            if not FP8GlobalStateManager.is_fp8_enabled():
                return

            weight_requires_grad = requires_grad and self.weight.requires_grad
            columnwise_usage = weight_requires_grad
            if FP8GlobalStateManager.get_fp8_recipe().backward_override is not None:
                columnwise_usage = False
            # The persistent weight cache must be a superset of all forward usages.
            # Validation forwards are rowwise-only, but training immediately needs
            # columnwise metadata for backward.
            self._fp4_debug_weight_cache_rowwise_usage = True
            self._fp4_debug_weight_cache_columnwise_usage = True

            input_quantizer = self.get_quantizer("forward", 0)
            weight_quantizer = self.get_quantizer("forward", 1)
            grad_output_quantizer = self.get_quantizer("backward", 0)
            input_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)
            weight_quantizer.set_usage(rowwise=True, columnwise=False)
            grad_output_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)

        def reset_recipe_state(self, *, recipe: Optional[Any]) -> None:
            super().reset_recipe_state(recipe=recipe)

            input_quantizer = self.get_quantizer("forward", 0)
            weight_quantizer = self.get_quantizer("forward", 1)
            grad_output_quantizer = self.get_quantizer("backward", 0)

            if input_quantizer is not None:
                input_quantizer.internal = True
                if not (self.tensor_parallel_mode == "column" and self.sequence_parallel):
                    input_quantizer.optimize_for_gemm = True
            if grad_output_quantizer is not None:
                grad_output_quantizer.internal = True
                if not (self.tensor_parallel_mode == "row" and self.sequence_parallel):
                    grad_output_quantizer.optimize_for_gemm = True
            if weight_quantizer is not None:
                weight_quantizer.internal = True

            self._fp4_debug_weight_cache_refresh_enabled = (
                recipe is not None and recipe.nvfp4()
            )

            if recipe is not None and recipe.nvfp4() and self.sequence_parallel:
                if self.tensor_parallel_mode == "column":
                    input_quantizer.with_amax_reduction = True
                    input_quantizer.amax_reduction_group = self.tensor_parallel_group
                elif self.tensor_parallel_mode == "row":
                    grad_output_quantizer.with_amax_reduction = True
                    grad_output_quantizer.amax_reduction_group = self.tensor_parallel_group

        def _get_or_update_fp4_debug_weight_cache(
            self,
            weight: torch.Tensor,
            weight_quantizer: Any,
            *,
            rowwise_usage: bool,
            columnwise_usage: bool,
        ) -> Optional[torch.Tensor]:
            if (
                weight_quantizer is None
                or not hasattr(weight_quantizer, "make_empty")
                or not hasattr(weight_quantizer, "update_quantized")
            ):
                return None

            cache = self._fp4_debug_weight_cache
            cache_matches = (
                cache is not None
                and is_te_quantized_tensor(cache)
                and tuple(cache.size()) == tuple(weight.size())
                and cache.dtype == weight.dtype
                and cache.device == weight.device
            )
            if cache_matches and hasattr(cache, "get_usages"):
                usages = cache.get_usages()
                cache_matches = (
                    usages.get("rowwise") == rowwise_usage
                    and usages.get("columnwise") == columnwise_usage
                )

            if not cache_matches:
                cache = weight_quantizer.make_empty(
                    weight.size(),
                    dtype=weight.dtype,
                    device=weight.device,
                    requires_grad=False,
                )
                self._fp4_debug_weight_cache = cache

            with torch.no_grad():
                weight_quantizer.update_quantized(weight.detach(), cache)

            return cache

        @staticmethod
        def _use_persistent_weight_shadow() -> bool:
            return os.getenv("FP4_MEGATRON_WEIGHT_SHADOW", "0").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )

        @staticmethod
        def _use_persistent_weight_shadow_in_forward() -> bool:
            return os.getenv("FP4_MEGATRON_WEIGHT_SHADOW_USE_IN_FORWARD", "0").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )

        @staticmethod
        def _keep_weight_cache_with_persistent_shadow() -> bool:
            return os.getenv("FP4_MEGATRON_WEIGHT_SHADOW_KEEP_CACHE_FOR_COMPARE", "0").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )

        def _shadow_is_primary_forward_weight(self) -> bool:
            return self._use_persistent_weight_shadow() and self._use_persistent_weight_shadow_in_forward()

        def _should_refresh_fp4_debug_weight_cache(self) -> bool:
            return (
                not self._shadow_is_primary_forward_weight()
                or self._keep_weight_cache_with_persistent_shadow()
            )

        @staticmethod
        def _skip_param_data_copy_after_all_gather_enabled() -> bool:
            return os.getenv(
                "FP4_MEGATRON_WEIGHT_SHADOW_SKIP_PARAM_DATA_COPY", "0"
            ).lower() in (
                "1",
                "true",
                "yes",
                "on",
            )

        def _skip_param_data_copy_after_all_gather(self) -> bool:
            return (
                self._skip_param_data_copy_after_all_gather_enabled()
                and self._shadow_is_primary_forward_weight()
                and self._persistent_weight_shadow_is_usable_for_forward(
                    self.weight,
                    rowwise_usage=True,
                    columnwise_usage=True,
                )
            )

        @staticmethod
        def _unmap_param_data_from_ddp_buffer_enabled() -> bool:
            return os.getenv(
                "FP4_MEGATRON_WEIGHT_SHADOW_UNMAP_PARAM_DATA", "0"
            ).lower() in (
                "1",
                "true",
                "yes",
                "on",
            )

        @staticmethod
        def _drop_persistent_param_data_enabled() -> bool:
            return os.getenv(
                "FP4_MEGATRON_WEIGHT_SHADOW_DROP_PERSISTENT_PARAM_DATA", "0"
            ).lower() in (
                "1",
                "true",
                "yes",
                "on",
            )

        def _unmap_param_data_from_ddp_buffer(self) -> bool:
            return (
                (
                    self._unmap_param_data_from_ddp_buffer_enabled()
                    or self._drop_persistent_param_data_enabled()
                )
                and self._shadow_is_primary_forward_weight()
            )

        def _drop_persistent_param_data(self) -> bool:
            return (
                self._drop_persistent_param_data_enabled()
                and self._shadow_is_primary_forward_weight()
            )

        def _get_or_update_fp4_persistent_weight_shadow(
            self,
            weight: torch.Tensor,
            weight_quantizer: Any,
            *,
            rowwise_usage: bool,
            columnwise_usage: bool,
        ) -> Optional[torch.Tensor]:
            if (
                not self._use_persistent_weight_shadow()
                or weight_quantizer is None
                or not hasattr(weight_quantizer, "make_empty")
                or not hasattr(weight_quantizer, "update_quantized")
            ):
                return None

            shadow = self._fp4_persistent_weight_shadow
            shadow_matches = (
                shadow is not None
                and is_te_quantized_tensor(shadow)
                and tuple(shadow.size()) == tuple(weight.size())
                and shadow.dtype == weight.dtype
                and shadow.device == weight.device
            )
            if shadow_matches and hasattr(shadow, "get_usages"):
                usages = shadow.get_usages()
                shadow_matches = (
                    usages.get("rowwise") == rowwise_usage
                    and usages.get("columnwise") == columnwise_usage
                )

            if not shadow_matches:
                shadow = weight_quantizer.make_empty(
                    weight.size(),
                    dtype=weight.dtype,
                    device=weight.device,
                    requires_grad=False,
                )
                self._fp4_persistent_weight_shadow = shadow

            with torch.no_grad():
                weight_quantizer.update_quantized(weight.detach(), shadow)

            return shadow

        def _refresh_fp4_debug_weight_cache_from_all_gather(
            self, gathered_weight: torch.Tensor
        ) -> None:
            if not self._fp4_debug_weight_cache_refresh_enabled:
                return
            weight_quantizer = self.get_quantizer("forward", 1)
            if weight_quantizer is None:
                return
            if not (
                hasattr(weight_quantizer, "make_empty")
                and hasattr(weight_quantizer, "update_quantized")
            ):
                return

            rowwise_usage = self._fp4_debug_weight_cache_rowwise_usage
            columnwise_usage = self._fp4_debug_weight_cache_columnwise_usage
            weight_quantizer.set_usage(rowwise=rowwise_usage, columnwise=columnwise_usage)
            ag_source = getattr(
                self.weight, "_fp4_megatron_weight_cache_ag_source", None
            )
            all_gather_source = (
                f"all_gather:{ag_source}" if ag_source is not None else "all_gather"
            )

            cache_refreshed = False
            if self._should_refresh_fp4_debug_weight_cache():
                self._get_or_update_fp4_debug_weight_cache(
                    gathered_weight,
                    weight_quantizer,
                    rowwise_usage=rowwise_usage,
                    columnwise_usage=columnwise_usage,
                )
                self._fp4_debug_weight_cache_source = all_gather_source
                cache_refreshed = True
            else:
                self._fp4_debug_weight_cache = None
                self._fp4_debug_weight_cache_source = "disabled:persistent_shadow"

            shadow = self._get_or_update_fp4_persistent_weight_shadow(
                gathered_weight,
                weight_quantizer,
                rowwise_usage=rowwise_usage,
                columnwise_usage=columnwise_usage,
            )
            if shadow is not None:
                self._fp4_persistent_weight_shadow_source = all_gather_source
                if cache_refreshed:
                    self._maybe_print_fp4_persistent_weight_shadow_compare()

        @staticmethod
        def _debug_rank() -> int:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return torch.distributed.get_rank()
            return 0

        @staticmethod
        def _is_current_stream_capturing() -> bool:
            try:
                return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
            except RuntimeError:
                return True

        @staticmethod
        def _use_weight_cache_in_forward() -> bool:
            return os.getenv("FP4_MEGATRON_WEIGHT_CACHE_USE_IN_FORWARD", "0").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )

        def _weight_cache_is_usable_for_forward(
            self,
            weight: torch.Tensor,
            *,
            rowwise_usage: bool,
            columnwise_usage: bool,
        ) -> bool:
            cache = self._fp4_debug_weight_cache
            if (
                cache is None
                or not is_te_quantized_tensor(cache)
                or tuple(cache.size()) != tuple(weight.size())
                or cache.dtype != weight.dtype
                or cache.device != weight.device
            ):
                return False
            if hasattr(cache, "get_usages"):
                usages = cache.get_usages()
                if rowwise_usage and not usages.get("rowwise"):
                    return False
                if columnwise_usage and not usages.get("columnwise"):
                    return False
            return True

        def _persistent_weight_shadow_is_usable_for_forward(
            self,
            weight: torch.Tensor,
            *,
            rowwise_usage: bool,
            columnwise_usage: bool,
        ) -> bool:
            shadow = self._fp4_persistent_weight_shadow
            if (
                shadow is None
                or not is_te_quantized_tensor(shadow)
                or tuple(shadow.size()) != tuple(weight.size())
                or shadow.dtype != weight.dtype
                or shadow.device != weight.device
            ):
                return False
            if hasattr(shadow, "get_usages"):
                usages = shadow.get_usages()
                if rowwise_usage and not usages.get("rowwise"):
                    return False
                if columnwise_usage and not usages.get("columnwise"):
                    return False
            return True

        @staticmethod
        def _metadata_tensors_match(lhs: Optional[torch.Tensor], rhs: Optional[torch.Tensor]):
            if lhs is None or rhs is None:
                if lhs is None and rhs is None:
                    return True, "none"
                if lhs is None:
                    return False, "forward-none/cache-present"
                return False, "forward-present/cache-none"
            if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype or lhs.device != rhs.device:
                return False, f"shape/dtype/device {lhs.shape}/{lhs.dtype}/{lhs.device} vs {rhs.shape}/{rhs.dtype}/{rhs.device}"
            equal = torch.equal(lhs, rhs)
            if equal:
                return True, f"{tuple(lhs.shape)} {lhs.dtype}"
            mismatch_count = torch.count_nonzero(lhs != rhs).item()
            if lhs.is_floating_point():
                max_abs = (lhs - rhs).abs().max().item()
            else:
                max_abs = (lhs.to(torch.int16) - rhs.to(torch.int16)).abs().max().item()
            return False, f"{tuple(lhs.shape)} {lhs.dtype} mismatches={mismatch_count} max_abs={max_abs}"

        def _make_fp4_debug_weight_cache_compare_message(
            self,
            forward_weight: Optional[torch.Tensor],
            cached_weight: Optional[torch.Tensor],
        ) -> Optional[str]:
            if self._is_current_stream_capturing():
                return self._make_fp4_debug_weight_cache_skip_message("stream_capturing")

            if forward_weight is None or cached_weight is None:
                max_missing_prints = int(
                    os.getenv("FP4_MEGATRON_WEIGHT_CACHE_MISSING_PRINTS", "0")
                )
                if (
                    max_missing_prints <= 0
                    or self._fp4_debug_weight_cache_missing_prints >= max_missing_prints
                ):
                    return self._make_fp4_debug_weight_cache_skip_message(
                        "missing_forward_or_cache"
                    )
                self._fp4_debug_weight_cache_missing_prints += 1
                return (
                    f"[FP4_WEIGHT_CACHE_COMPARE] rank={self._debug_rank()} "
                    f"op={id(self)} shape={tuple(self.weight.shape)} "
                    f"cache_source={self._fp4_debug_weight_cache_source} "
                    f"weight_source={self._fp4_debug_weight_compute_source} "
                    f"forward_weight={'set' if forward_weight is not None else 'none'} "
                    f"cache={'set' if cached_weight is not None else 'none'}"
                )

            max_prints = int(os.getenv("FP4_MEGATRON_WEIGHT_CACHE_COMPARE_PRINTS", "0"))
            if max_prints <= 0 or self._fp4_debug_weight_cache_compare_prints >= max_prints:
                return self._make_fp4_debug_weight_cache_skip_message("compare_budget_exhausted")
            self._fp4_debug_weight_cache_compare_prints += 1
            print_budget = f"{self._fp4_debug_weight_cache_compare_prints}/{max_prints}"

            if not (
                is_te_quantized_tensor(forward_weight)
                and is_te_quantized_tensor(cached_weight)
                and hasattr(forward_weight, "get_metadata")
                and hasattr(cached_weight, "get_metadata")
            ):
                return (
                    f"[FP4_WEIGHT_CACHE_COMPARE] rank={self._debug_rank()} "
                    f"op={id(self)} shape={tuple(self.weight.shape)} "
                    f"cache_source={self._fp4_debug_weight_cache_source} "
                    f"weight_source={self._fp4_debug_weight_compute_source} "
                    f"print_budget={print_budget} "
                    f"forward_quantized={is_te_quantized_tensor(forward_weight)} "
                    f"cache_quantized={is_te_quantized_tensor(cached_weight)}"
                )

            forward_metadata = forward_weight.get_metadata()
            cache_metadata = cached_weight.get_metadata()
            keys = (
                "rowwise_data",
                "rowwise_scale_inv",
                "columnwise_data",
                "columnwise_scale_inv",
                "amax_rowwise",
                "amax_columnwise",
            )
            required_keys = {
                key for key in keys if forward_metadata.get(key) is not None
            }
            parts = []
            required_match = True
            exact_metadata_match = True
            for key in keys:
                match, detail = self._metadata_tensors_match(
                    forward_metadata.get(key), cache_metadata.get(key)
                )
                exact_metadata_match = exact_metadata_match and match
                if key in required_keys:
                    required_match = required_match and match
                    parts.append(f"{key}={match}({detail})")
                elif match:
                    parts.append(f"{key}=True({detail})")
                else:
                    parts.append(f"{key}=extra({detail})")

            return (
                f"[FP4_WEIGHT_CACHE_COMPARE] rank={self._debug_rank()} op={id(self)} "
                f"shape={tuple(self.weight.shape)} "
                f"cache_source={self._fp4_debug_weight_cache_source} "
                f"weight_source={self._fp4_debug_weight_compute_source} "
                f"print_budget={print_budget} "
                f"required_match={required_match} exact_metadata_match={exact_metadata_match} "
                + " ".join(parts)
            )

        def _maybe_print_fp4_persistent_weight_shadow_compare(self) -> None:
            if self._is_current_stream_capturing():
                return
            max_prints = int(os.getenv("FP4_MEGATRON_WEIGHT_SHADOW_COMPARE_PRINTS", "0"))
            if (
                max_prints <= 0
                or self._fp4_persistent_weight_shadow_compare_prints >= max_prints
            ):
                return
            self._fp4_persistent_weight_shadow_compare_prints += 1
            print_budget = (
                f"{self._fp4_persistent_weight_shadow_compare_prints}/{max_prints}"
            )

            cache = self._fp4_debug_weight_cache
            shadow = self._fp4_persistent_weight_shadow
            if not (
                is_te_quantized_tensor(cache)
                and is_te_quantized_tensor(shadow)
                and hasattr(cache, "get_metadata")
                and hasattr(shadow, "get_metadata")
            ):
                match = False
                detail = (
                    f"cache_quantized={is_te_quantized_tensor(cache)} "
                    f"shadow_quantized={is_te_quantized_tensor(shadow)}"
                )
            else:
                cache_metadata = cache.get_metadata()
                shadow_metadata = shadow.get_metadata()
                keys = (
                    "rowwise_data",
                    "rowwise_scale_inv",
                    "columnwise_data",
                    "columnwise_scale_inv",
                    "amax_rowwise",
                    "amax_columnwise",
                )
                required_keys = {
                    key for key in keys if cache_metadata.get(key) is not None
                }
                parts = []
                match = True
                exact_metadata_match = True
                for key in keys:
                    key_match, key_detail = self._metadata_tensors_match(
                        cache_metadata.get(key), shadow_metadata.get(key)
                    )
                    exact_metadata_match = exact_metadata_match and key_match
                    if key in required_keys:
                        match = match and key_match
                        parts.append(f"{key}={key_match}({key_detail})")
                    elif key_match:
                        parts.append(f"{key}=True({key_detail})")
                    else:
                        parts.append(f"{key}=extra({key_detail})")
                detail = (
                    f"required_match={match} "
                    f"exact_metadata_match={exact_metadata_match} "
                    + " ".join(parts)
                )

            print(
                f"[FP4_WEIGHT_SHADOW_COMPARE] rank={self._debug_rank()} "
                f"op={id(self)} shape={tuple(self.weight.shape)} "
                f"cache_source={self._fp4_debug_weight_cache_source} "
                f"shadow_source={self._fp4_persistent_weight_shadow_source} "
                f"print_budget={print_budget} match={match} {detail}",
                flush=True,
            )
            strict = os.getenv(
                "FP4_MEGATRON_WEIGHT_SHADOW_COMPARE_STRICT", "1"
            ).lower() in ("1", "true", "yes", "on")
            if strict and not match:
                raise RuntimeError(
                    "FP4 persistent weight shadow does not match the weight cache: "
                    f"shape={tuple(self.weight.shape)} {detail}"
                )

        def _make_fp4_debug_weight_cache_skip_message(self, reason: str) -> Optional[str]:
            max_skip_prints = int(os.getenv("FP4_MEGATRON_WEIGHT_CACHE_SKIP_PRINTS", "0"))
            if max_skip_prints <= 0 or self._fp4_debug_weight_cache_skip_prints >= max_skip_prints:
                return None
            self._fp4_debug_weight_cache_skip_prints += 1
            max_prints = os.getenv("FP4_MEGATRON_WEIGHT_CACHE_COMPARE_PRINTS", "0")
            return (
                f"[FP4_WEIGHT_CACHE_SKIP] rank={self._debug_rank()} op={id(self)} "
                f"shape={tuple(self.weight.shape)} reason={reason} "
                f"cache_source={self._fp4_debug_weight_cache_source} "
                f"weight_source={self._fp4_debug_weight_compute_source} "
                f"cache={'set' if self._fp4_debug_weight_cache is not None else 'none'} "
                f"compare_prints={self._fp4_debug_weight_cache_compare_prints}/{max_prints} "
                f"grad_enabled={torch.is_grad_enabled()} training={self.training}"
            )

        def _maybe_print_fp4_debug_weight_cache_trace(self) -> None:
            max_prints = int(os.getenv("FP4_MEGATRON_WEIGHT_CACHE_TRACE_PRINTS", "0"))
            if (
                max_prints <= 0
                or self._fp4_debug_weight_cache_trace_prints >= max_prints
            ):
                return
            self._fp4_debug_weight_cache_trace_prints += 1
            print(
                f"[FP4_WEIGHT_CACHE_TRACE] rank={self._debug_rank()} "
                f"op={id(self)} shape={tuple(self.weight.shape)} "
                f"weight_source={self._fp4_debug_weight_compute_source} "
                f"cache_source={self._fp4_debug_weight_cache_source} "
                f"shadow_source={self._fp4_persistent_weight_shadow_source} "
                f"cache={'set' if self._fp4_debug_weight_cache is not None else 'none'} "
                f"shadow={'set' if self._fp4_persistent_weight_shadow is not None else 'none'} "
                f"trace_prints={self._fp4_debug_weight_cache_trace_prints}/{max_prints} "
                f"grad_enabled={torch.is_grad_enabled()} training={self.training}",
                flush=True,
            )

        def _print_fp4_debug_pending_weight_cache_compare(self) -> None:
            try:
                if self._fp4_debug_pending_compare_message is not None:
                    print(self._fp4_debug_pending_compare_message, flush=True)
            finally:
                self._fp4_debug_pending_compare_message = None

        def op_forward(
            self,
            ctx,
            input_: torch.Tensor,
            prev_op_grad_output_quantizer: Optional[Any],
            next_op_input_quantizer: Optional[Any],
        ) -> torch.Tensor:
            input_requires_grad = ctx.requires_grad
            weight_requires_grad = ctx.requires_grad and self.weight.requires_grad

            input_quantizer = self.get_quantizer("forward", 0)
            weight_quantizer = self.get_quantizer("forward", 1)
            output_quantizer = next_op_input_quantizer
            grad_output_quantizer = self.get_quantizer("backward", 0)
            grad_input_quantizer = prev_op_grad_output_quantizer

            with_quantized_compute = FP8GlobalStateManager.is_fp8_enabled()
            if with_quantized_compute:
                backward_override = FP8GlobalStateManager.get_fp8_recipe().backward_override
            else:
                backward_override = None

            if torch.is_autocast_enabled():
                dtype = torch.get_autocast_dtype("cuda")
            else:
                dtype = self.weight.dtype

            compute_weight = self.weight
            using_prequantized_weight = False
            weight_columnwise_usage = False
            self._fp4_debug_weight_compute_source = "bf16"
            self._fp4_debug_pending_compare_message = None
            if not with_quantized_compute:
                self._fp4_debug_pending_compare_message = (
                    self._make_fp4_debug_weight_cache_skip_message(
                        "quantized_compute_disabled"
                    )
                )
            elif is_te_quantized_tensor(compute_weight):
                self._fp4_debug_pending_compare_message = (
                    self._make_fp4_debug_weight_cache_skip_message("weight_already_quantized")
                )
            else:
                if weight_quantizer is None:
                    raise ValueError("Missing quantizer for weight tensor")
                weight_columnwise_usage = input_requires_grad and backward_override is None
                self._fp4_debug_weight_cache_rowwise_usage = True
                self._fp4_debug_weight_cache_columnwise_usage = True
                weight_quantizer.set_usage(
                    rowwise=True,
                    columnwise=weight_columnwise_usage,
                )
                if (
                    self._use_persistent_weight_shadow_in_forward()
                    and self._persistent_weight_shadow_is_usable_for_forward(
                        compute_weight,
                        rowwise_usage=True,
                        columnwise_usage=weight_columnwise_usage,
                    )
                ):
                    compute_weight = self._fp4_persistent_weight_shadow
                    using_prequantized_weight = True
                    self._fp4_debug_weight_compute_source = "persistent_shadow"
                elif (
                    self._use_weight_cache_in_forward()
                    and self._weight_cache_is_usable_for_forward(
                        compute_weight,
                        rowwise_usage=True,
                        columnwise_usage=weight_columnwise_usage,
                    )
                ):
                    compute_weight = self._fp4_debug_weight_cache
                    using_prequantized_weight = True
                    self._fp4_debug_weight_compute_source = "cache"
                else:
                    compute_weight = weight_quantizer(compute_weight)
                    self._fp4_debug_weight_compute_source = "forward_quantizer"
                if self._should_refresh_fp4_debug_weight_cache():
                    self._fp4_debug_pending_compare_message = (
                        self._make_fp4_debug_weight_cache_compare_message(
                            compute_weight,
                            self._fp4_debug_weight_cache,
                        )
                    )
            self._maybe_print_fp4_debug_weight_cache_trace()

            output, x_local, w = te.pytorch.ops.BasicLinear._functional_forward(
                input=input_,
                weight=compute_weight,
                dtype=dtype,
                tensor_parallel_mode=self.tensor_parallel_mode,
                tensor_parallel_group=self.tensor_parallel_group,
                sequence_parallel=self.sequence_parallel,
                with_quantized_compute=with_quantized_compute,
                backward_override=backward_override,
                input_quantizer=input_quantizer,
                weight_quantizer=weight_quantizer,
                output_quantizer=output_quantizer,
                input_requires_grad=input_requires_grad,
                weight_requires_grad=weight_requires_grad,
            )

            if (
                input_requires_grad
                and compute_weight is not self.weight
                and with_quantized_compute
                and is_te_quantized_tensor(w)
                and backward_override is None
            ):
                if not using_prequantized_weight:
                    w.update_usage(rowwise_usage=False, columnwise_usage=True)

            if ctx.requires_grad:
                if backward_override == "high_precision":
                    saved_input = input_ if weight_requires_grad else None
                    saved_weight = self.weight if input_requires_grad else None
                else:
                    saved_input = x_local
                    saved_weight = w
                if is_cpu_offload_enabled():
                    mark_activation_offload(saved_input)
                ctx.save_for_backward(saved_input, saved_weight)
                ctx.with_quantized_compute = with_quantized_compute and backward_override is None
                ctx.backward_override = backward_override
                ctx.input_quantizer = input_quantizer
                ctx.weight_quantizer = weight_quantizer
                ctx.grad_output_quantizer = grad_output_quantizer
                ctx.grad_input_quantizer = grad_input_quantizer
                ctx.dtype = dtype
                ctx.input_requires_grad = input_requires_grad
                ctx.weight_requires_grad = weight_requires_grad

            return output

        def op_backward(self, ctx, grad_output: torch.Tensor):
            return te.pytorch.ops.BasicLinear.op_backward(self, ctx, grad_output)

    class TEFusedMLP(MLP):
        """MLP wrapper using Transformer Engine's operation-based API."""

        @copy_signature(MLP.__init__)
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Fused implementation
            self._fused_impl: Optional[Tuple[te.pytorch.ops.Sequential]] = None
            self._register_fp4_weight_param_pre_ddp_hooks()

        def _register_fp4_weight_param_pre_ddp_hooks(self) -> None:
            if not self.config.fp4_megatron_weight_quantization:
                return

            def unmap_param_data_from_ddp_buffer() -> bool:
                return (
                    (
                        _MegatronQuantizedBasicLinear._unmap_param_data_from_ddp_buffer_enabled()
                        or _MegatronQuantizedBasicLinear._drop_persistent_param_data_enabled()
                    )
                    and _MegatronQuantizedBasicLinear._use_persistent_weight_shadow()
                    and _MegatronQuantizedBasicLinear._use_persistent_weight_shadow_in_forward()
                )

            def drop_persistent_param_data() -> bool:
                return (
                    _MegatronQuantizedBasicLinear._drop_persistent_param_data_enabled()
                    and _MegatronQuantizedBasicLinear._use_persistent_weight_shadow()
                    and _MegatronQuantizedBasicLinear._use_persistent_weight_shadow_in_forward()
                )

            for weight in (self.linear_fc1.weight, self.linear_fc2.weight):
                weight._fp4_megatron_unmap_param_data_from_ddp_buffer = (
                    unmap_param_data_from_ddp_buffer
                )
                weight._fp4_megatron_drop_persistent_param_data = (
                    drop_persistent_param_data
                )

        def _make_fused_impl(self) -> te.pytorch.ops.Sequential:
            """Construct fused module matching MLP."""

            # Container for fusible ops
            fused_impl = te.pytorch.ops.Sequential()
            basic_linear_cls = (
                _MegatronQuantizedBasicLinear
                if self.config.fp4_megatron_weight_quantization
                else te.pytorch.ops.BasicLinear
            )

            # Tensor parallelism configuration
            tp_world_size = get_tensor_model_parallel_world_size()
            tp_group = None
            if tp_world_size > 1:
                tp_group = get_tensor_model_parallel_group()

            # RNG state
            rng_state_tracker_function = None
            if get_cuda_rng_tracker().is_initialized():
                rng_state_tracker_function = get_cuda_rng_tracker

            # Check submodule types
            if not isinstance(self.linear_fc1, te.pytorch.LayerNormLinear):
                raise ValueError(
                    f"{self.__class__.__name__} expects FC1 to be "
                    "Transformer Engine LayerNormLinear, but found "
                    f"{self.linear_fc1.__class__.__name__}."
                )
            if not isinstance(self.linear_fc2, te.pytorch.Linear):
                raise ValueError(
                    f"{self.__class__.__name__} expects FC1 to be "
                    "Transformer Engine Linear, but found "
                    f"{self.linear_fc2.__class__.__name__}."
                )

            # Norm op
            norm_type = self.linear_fc1.normalization
            norm_shape = self.linear_fc1.weight.size(1)
            kwargs = {
                "eps": self.linear_fc1.eps,
                "device": "meta",
                "dtype": self.linear_fc1.layer_norm_weight.dtype,
                "zero_centered_gamma": self.linear_fc1.zero_centered_gamma,
            }
            op = None
            if norm_type == "LayerNorm":
                op = te.pytorch.ops.LayerNorm(norm_shape, **kwargs)
                op.weight = self.linear_fc1.layer_norm_weight
                op.bias = self.linear_fc1.layer_norm_bias
            elif norm_type == "RMSNorm":
                op = te.pytorch.ops.RMSNorm(norm_shape, **kwargs)
                op.weight = self.linear_fc1.layer_norm_weight
            else:
                raise ValueError(f"Unsupported normalization ({norm_type})")
            fused_impl.append(op)

            # FC1 linear op
            weight = self.linear_fc1.weight
            userbuffers_options = None
            if self.linear_fc1.config.tp_comm_overlap and self.linear_fc1.ub_name is not None:
                userbuffers_options = {"comm_name": self.linear_fc1.ub_name}
            op = basic_linear_cls(
                weight.size(1),
                weight.size(0) * tp_world_size,
                device="meta",
                dtype=weight.dtype,
                tensor_parallel_mode="column" if tp_world_size > 1 else None,
                tensor_parallel_group=tp_group,
                sequence_parallel=self.linear_fc1.sequence_parallel,
                rng_state_tracker_function=rng_state_tracker_function,
                accumulate_into_main_grad=self.linear_fc1.fuse_wgrad_accumulation,
                userbuffers_options=userbuffers_options,
            )
            op.weight = weight
            if hasattr(op, "_register_fp4_weight_cache_refresh_hook"):
                op._register_fp4_weight_cache_refresh_hook()
            fused_impl.append(op)

            # FC1 bias op
            bias = self.linear_fc1.bias
            if isinstance(bias, torch.Tensor) and bias.numel() == 0:
                bias = None
            if bias is not None:
                op = te.pytorch.ops.Bias(bias.numel(), device="meta", dtype=bias.dtype)
                op.bias = bias
                fused_impl.append(op)

            # Activation op
            op = self._make_activation_op(
                self.activation_func,
                self.config.gated_linear_unit,
                self.config.activation_func_fp8_input_store,
            )
            fused_impl.append(op)

            # FC2 linear op
            weight = self.linear_fc2.weight
            userbuffers_options = None
            if self.linear_fc2.config.tp_comm_overlap and self.linear_fc2.ub_name is not None:
                userbuffers_options = {"comm_name": self.linear_fc2.ub_name}
            op = basic_linear_cls(
                weight.size(1),
                weight.size(0),
                device="meta",
                dtype=weight.dtype,
                rng_state_tracker_function=rng_state_tracker_function,
                accumulate_into_main_grad=self.linear_fc2.fuse_wgrad_accumulation,
                userbuffers_options=userbuffers_options,
            )
            op.weight = weight
            if hasattr(op, "_register_fp4_weight_cache_refresh_hook"):
                op._register_fp4_weight_cache_refresh_hook()
            fused_impl.append(op)
            if tp_world_size > 1:
                if self.linear_fc2.sequence_parallel:
                    fused_impl.append(te.pytorch.ops.ReduceScatter(tp_group))
                else:
                    fused_impl.append(te.pytorch.ops.AllReduce(tp_group))

            # FC2 bias op
            if not self.linear_fc2.te_return_bias:
                bias = self.linear_fc2.bias
                if isinstance(bias, torch.Tensor) and bias.numel() == 0:
                    bias = None
                if bias is not None:
                    op = te.pytorch.ops.Bias(bias.numel(), device="meta", dtype=bias.dtype)
                    op.bias = bias
                    fused_impl.append(op)

            # Emulate submodule forward hooks if needed
            self._register_hooks_on_fused_impl(fused_impl)

            return fused_impl

        def _make_activation_op(
            self, activation_func: Callable, gated_linear_unit: bool, cache_quantized_input: bool
        ) -> te.pytorch.ops.FusibleOperation:
            """Construct activation op."""

            # Get op type
            op_type = None
            if (activation_func, gated_linear_unit) == (F.gelu, False):
                op_type = te.pytorch.ops.GELU
            elif (activation_func, gated_linear_unit) == (F.gelu, True):
                op_type = te.pytorch.ops.GEGLU
            elif (activation_func, gated_linear_unit) == (F.silu, False):
                if not is_te_min_version("2.8.0"):
                    raise NotImplementedError("SiLU activation requires Transformer Engine 2.8+")
                op_type = te.pytorch.ops.SiLU
            elif (activation_func, gated_linear_unit) == (F.silu, True):
                op_type = te.pytorch.ops.SwiGLU
            elif (activation_func, gated_linear_unit) == (F.relu, False):
                op_type = te.pytorch.ops.ReLU
            elif (activation_func, gated_linear_unit) == (F.relu, True):
                op_type = te.pytorch.ops.ReGLU

            # Could not find corresponding activation op
            if op_type is None:
                raise NotImplementedError(
                    "Transformer Engine operation-based API does not support "
                    f"activation_func={activation_func}, "
                    f"gated_linear_unit={gated_linear_unit}"
                )

            # Construct op
            kwargs = {}
            if is_te_min_version("2.3"):
                kwargs["cache_quantized_input"] = cache_quantized_input
            return op_type(**kwargs)

        def _register_hooks_on_fused_impl(self, fused_impl: torch.nn.Module) -> None:
            """Attempt to emulate submodule callback hooks.

            This is not always possible because Transformer Engine's
            op fuser does not expose intermediate tensors. Depending
            on what kernel fusions the op fuser chooses, the
            intermediate tensors may not even exist. Hooks that modify
            tensors will result in incorrect behavior.

            """

            # Get submodule hooks
            forward_pre_hooks = []
            forward_post_hooks = []
            backward_pre_hooks = []
            backward_post_hooks = []
            for submodule in self.modules():
                for hook_id, hook in submodule._forward_pre_hooks.items():
                    with_kwargs = hook_id in submodule._forward_pre_hooks_with_kwargs
                    forward_pre_hooks.append((submodule, hook, with_kwargs))
                for hook_id, hook in submodule._forward_hooks.items():
                    with_kwargs = hook_id in submodule._forward_hooks_with_kwargs
                    forward_post_hooks.append((submodule, hook, with_kwargs))
                for hook in submodule._backward_pre_hooks.values():
                    backward_pre_hooks.append((submodule, hook))
                for hook in submodule._backward_hooks.values():
                    backward_post_hooks.append((submodule, hook))

            # Pre-forward hooks
            # Note: DDP pre-forward hooks are safe since they do not
            # interact with input tensor.
            if forward_pre_hooks:
                from megatron.core.distributed import distributed_data_parallel

                if any(
                    inspect.getmodule(hook) != distributed_data_parallel
                    for _, hook, _ in forward_pre_hooks
                ):
                    warnings.warn(
                        "TEFusedMLP module has a submodule with a pre-forward hook. "
                        "TEFusedMLP module does not expose intermediate tensors, "
                        "so the hook may have incorrect behavior if it attempts to "
                        "access the input tensor."
                    )

                def forward_pre_hook(module, *_) -> None:
                    for submodule, hook, with_kwargs in forward_pre_hooks:
                        if with_kwargs:
                            ret = hook(submodule, (), {})
                        else:
                            ret = hook(submodule, ())
                        if ret is not None:
                            raise RuntimeError(
                                "TEFusedMLP module does not expose intermediate tensors, but "
                                "submodule has pre-forward hook that modifies input tensor."
                            )

                fused_impl.register_forward_pre_hook(forward_pre_hook)

            # Post-forward hooks
            if forward_post_hooks:
                warnings.warn(
                    "TEFusedMLP module has a submodule with a post-forward hook. "
                    "TEFusedMLP module does not expose intermediate tensors, "
                    "so the hook may have incorrect behavior if it attempts to "
                    "access the input or output tensors."
                )

                def forward_post_hook(module, *_) -> None:
                    for submodule, hook, with_kwargs in forward_post_hooks:
                        if with_kwargs:
                            ret = hook(submodule, (), {}, None)
                        else:
                            ret = hook(submodule, (), None)
                        if ret is not None:
                            raise RuntimeError(
                                "TEFusedMLP module does not expose intermediate tensors, but "
                                "submodule has post-forward hook that modifies output tensor."
                            )

                fused_impl.register_forward_hook(forward_post_hook)

            # Backward hooks
            if backward_pre_hooks:
                raise RuntimeError(
                    "TEFusedMLP module does not support submodules with pre-backward hooks"
                )
            if backward_post_hooks:
                raise RuntimeError(
                    "TEFusedMLP module does not support submodules with post-backward hooks"
                )

        def forward(self, hidden_states: torch.Tensor, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
            """Forward."""

            # Construct fused impl if needed
            # Note: We initialize during the first forward pass in
            # case the params are modified after the constructor.
            # Note: The fused impl is stored in a tuple to avoid
            # registering as a submodule.
            if self._fused_impl is None:
                self._fused_impl = (self._make_fused_impl(),)

            # Apply fused impl
            out = self._fused_impl[0](hidden_states)
            if self.config.fp4_megatron_weight_quantization:
                for op in self._fused_impl[0]:
                    print_compare = getattr(
                        op, "_print_fp4_debug_pending_weight_cache_compare", None
                    )
                    if print_compare is not None:
                        print_compare()

            # Return bias tensor if requested
            bias = None
            if self.linear_fc2.te_return_bias:
                bias = self.linear_fc2.bias
                if isinstance(bias, torch.Tensor) and bias.numel() == 0:
                    bias = None

            return out, bias

    class TEFusedDenseMLP(TEFusedMLP):
        """Dense MLP using GroupedLinear(num_groups=1) to trigger
        ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8 fusion on SM100+ with MXFP8 recipe.

        Subclass of TEFusedMLP -> does not modify TEFusedMLP or TEGroupedMLP.
        The fused kernel fires automatically via the TE op fuser when it detects
        the GroupedLinear -> ScaledSwiGLU -> GroupedLinear pattern with MXFP8 recipe.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._norm_seq: Optional[Tuple[te.pytorch.ops.Sequential]] = None
            if not is_te_min_version("2.14.0"):
                raise RuntimeError(
                    f"{self.__class__.__name__} requires Transformer Engine >= 2.14.0 "
                    "(needs pytorch.ops.GroupedLinear and pytorch.ops.ScaledSwiGLU)"
                )
            if self.config.add_bias_linear:
                raise ValueError(
                    f"{self.__class__.__name__} does not support add_bias_linear=True; "
                    "the CuTeGEMM fused kernel requires bias-free linear layers."
                )
            if self.config.activation_func != F.silu or not self.config.gated_linear_unit:
                raise ValueError(
                    f"{self.__class__.__name__} requires SwiGLU activation "
                    "(activation_func=F.silu, gated_linear_unit=True) "
                    "for the CuTeGEMM fused kernel, but got "
                    f"activation_func={self.config.activation_func}, "
                    f"gated_linear_unit={self.config.gated_linear_unit}."
                )

        def _make_fused_impl(self) -> te.pytorch.ops.Sequential:
            """Construct fused module with GroupedLinear(num_groups=1) + ScaledSwiGLU."""

            fused_impl = te.pytorch.ops.Sequential()

            # Tensor parallelism configuration
            tp_world_size = get_tensor_model_parallel_world_size()
            tp_group = None
            if tp_world_size > 1:
                tp_group = get_tensor_model_parallel_group()

            # RNG state
            rng_state_tracker_function = None
            if get_cuda_rng_tracker().is_initialized():
                rng_state_tracker_function = get_cuda_rng_tracker

            # Check submodule types (same as TEFusedMLP)
            if not isinstance(self.linear_fc1, te.pytorch.LayerNormLinear):
                raise ValueError(
                    f"{self.__class__.__name__} expects FC1 to be "
                    "Transformer Engine LayerNormLinear, but found "
                    f"{self.linear_fc1.__class__.__name__}."
                )
            if not isinstance(self.linear_fc2, te.pytorch.Linear):
                raise ValueError(
                    f"{self.__class__.__name__} expects FC2 to be "
                    "Transformer Engine Linear, but found "
                    f"{self.linear_fc2.__class__.__name__}."
                )

            # Norm op (same as TEFusedMLP)
            norm_type = self.linear_fc1.normalization
            norm_shape = self.linear_fc1.weight.size(1)
            kwargs = {
                "eps": self.linear_fc1.eps,
                "device": "meta",
                "dtype": self.linear_fc1.layer_norm_weight.dtype,
                "zero_centered_gamma": self.linear_fc1.zero_centered_gamma,
            }
            op = None
            if norm_type == "LayerNorm":
                op = te.pytorch.ops.LayerNorm(norm_shape, **kwargs)
                op.weight = self.linear_fc1.layer_norm_weight
                op.bias = self.linear_fc1.layer_norm_bias
            elif norm_type == "RMSNorm":
                op = te.pytorch.ops.RMSNorm(norm_shape, **kwargs)
                op.weight = self.linear_fc1.layer_norm_weight
            else:
                raise ValueError(f"Unsupported normalization ({norm_type})")
            # Store norm in a separate Sequential applied OUTSIDE the MXFP8 autocast
            # in forward(). Running norm inside MXFP8 context corrupts the saved rstd
            # used in RMSNorm backward, causing gradient amplification up to 10^6.
            # Wrapped in tuple to avoid nn.Module submodule registration (which would
            # duplicate the shared norm weight in state_dict/parameters).
            norm_seq = te.pytorch.ops.Sequential()
            norm_seq.append(op)
            self._norm_seq = (norm_seq,)

            # GLU interleave size must match ScaledSwiGLU and the CuTe kernel.
            _GLU_INTERLEAVE_SIZE = 32

            # FC1: GroupedLinear(num_groups=1) instead of BasicLinear
            weight = self.linear_fc1.weight
            op = te.pytorch.ops.GroupedLinear(
                num_groups=1,
                in_features=weight.size(1),
                out_features=weight.size(0) * tp_world_size,
                device="meta",
                dtype=weight.dtype,
                bias=False,
                rng_state_tracker_function=rng_state_tracker_function,
                accumulate_into_main_grad=self.linear_fc1.fuse_wgrad_accumulation,
            )
            op.weight0 = weight
            op._glu_interleave_size = _GLU_INTERLEAVE_SIZE  # signals fuser_forward to interleave
            fused_impl.append(op)

            # ScaledSwiGLU with glu_interleave_size=32
            # Required by ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8
            fused_impl.append(te.pytorch.ops.ScaledSwiGLU(glu_interleave_size=32))

            # FC2: GroupedLinear(num_groups=1) instead of BasicLinear
            weight = self.linear_fc2.weight
            op = te.pytorch.ops.GroupedLinear(
                num_groups=1,
                in_features=weight.size(1),
                out_features=weight.size(0),
                device="meta",
                dtype=weight.dtype,
                bias=False,
                rng_state_tracker_function=rng_state_tracker_function,
                accumulate_into_main_grad=self.linear_fc2.fuse_wgrad_accumulation,
            )
            op.weight0 = weight
            # FC2 has no SwiGLU — MXFP8 quantization done on-the-fly in fuser_forward.
            # No _mxfp8_weight0 pre-computation to avoid ~28 GB persistent FP8 tensors.
            fused_impl.append(op)

            if tp_world_size > 1:
                if self.linear_fc2.sequence_parallel:
                    fused_impl.append(te.pytorch.ops.ReduceScatter(tp_group))
                else:
                    fused_impl.append(te.pytorch.ops.AllReduce(tp_group))

            self._register_hooks_on_fused_impl(fused_impl)
            return fused_impl

        def forward(self, hidden_states: torch.Tensor, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
            """Forward pass using GroupedLinear(num_groups=1) + ScaledSwiGLU."""

            orig_shape = hidden_states.shape
            hidden_size = hidden_states.size(-1)
            hidden_states_2d = hidden_states.view(-1, hidden_size)
            total_tokens = hidden_states_2d.size(0)

            tokens_per_expert = torch.full(
                (1,), total_tokens, dtype=torch.long, device=hidden_states.device
            )
            scales = torch.ones(
                total_tokens, device=hidden_states.device, dtype=hidden_states.dtype
            )

            # Build fused impl and cache recipe lazily on first forward pass.
            # Both are created once and reused — avoids object creation every call.
            if not hasattr(self, '_recipe'):
                if os.getenv("FP4_RECIPE", "") == "nvfp4":
                    self._recipe = te.common.recipe.NVFP4BlockScaling()
                else:
                    self._recipe = te.common.recipe.MXFP8BlockScaling()
            recipe = self._recipe

            if self._fused_impl is None:
                with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=recipe):
                    self._fused_impl = (self._make_fused_impl(),)

            # Apply norm in BF16 OUTSIDE the MXFP8 autocast to preserve the rstd
            # tensor used by RMSNorm backward (running it inside causes up to 10^6
            # gradient amplification, and causes convergence issues).
            normed = self._norm_seq[0](hidden_states_2d)

            with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=recipe):
                out = self._fused_impl[0](normed, tokens_per_expert, scales, tokens_per_expert)

            out = out.view(*orig_shape[:-1], out.size(-1))

            bias = None
            if self.linear_fc2.te_return_bias:
                bias = self.linear_fc2.bias
                if isinstance(bias, torch.Tensor) and bias.numel() == 0:
                    bias = None

            return out, bias

else:
    TEFusedMLP = None  # type: ignore[assignment, misc]
    TEFusedDenseMLP = None  # type: ignore[assignment, misc]


class TEDelayedScaling(te.common.recipe.DelayedScaling):
    """
    Wrapper for the Transformer-Engine's `DelayedScaling` layer.
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        fp8_format: int,
        override_linear_precision: tuple = (False, False, False),
    ):
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        extra_kwargs = _get_extra_te_kwargs(config)
        if is_te_min_version("1.6.0.dev0"):
            extra_kwargs["fp8_dpa"] = config.fp8_dot_product_attention
            extra_kwargs["fp8_mha"] = config.fp8_multi_head_attention
        if get_te_version() < PkgVersion("1.8.0"):
            extra_kwargs["interval"] = config.fp8_interval
        elif config.fp8_interval != 1:
            warnings.warn("fp8_interval is deprecated and ignored from Transformer-Engine v1.8.0.")

        super().__init__(
            margin=config.fp8_margin,
            fp8_format=fp8_format,
            amax_compute_algo=config.fp8_amax_compute_algo,
            amax_history_len=config.fp8_amax_history_len,
            override_linear_precision=override_linear_precision,
            **extra_kwargs,
        )


class TECudaRNGStatesTracker(te.pytorch.distributed.CudaRNGStatesTracker):
    """Wraps TransformerEngine's CudaRNGStatesTracker so that it is
    interchangeable with Megatron's RNG tracker"""

    def __init__(self, is_inference_rng_tracker=False):
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        super().__init__()
        self.reset()
        self.is_inference_rng_tracker = is_inference_rng_tracker

    def is_initialized(self):
        """Checks if the internal RNG state has been set with set_states()."""
        return self._is_initialized

    def reset(self):
        """Reset the internal RNG state."""
        super().reset()
        self._is_initialized = False

    def set_states(self, states):
        """Set the internal RNG state."""
        super().set_states(states)
        self._is_initialized = True

    def add(self, name, seed):
        """Track the rng state."""
        super().add(name, seed)
        self._is_initialized = True


def te_checkpoint(
    forward_func, distribute_saved_activations, get_rng_state_tracker, tp_group, *args, **kwargs
):
    """Checkpointing with Transformer-Engine."""
    if not HAVE_TE:
        raise ImportError(
            "Transformer Engine is not installed. "
            "Please install it with `pip install transformer-engine`."
        )

    from transformer_engine.pytorch.distributed import checkpoint

    if is_te_min_version("1.5.0"):
        return checkpoint(
            forward_func,
            *args,
            distribute_saved_activations=distribute_saved_activations,
            get_rng_state_tracker=get_rng_state_tracker,
            tp_group=tp_group,
            **kwargs,
        )
    else:
        return checkpoint(
            forward_func, distribute_saved_activations, get_rng_state_tracker, tp_group, *args
        )


try:
    from transformer_engine.pytorch.attention import _SplitAlongDim

    SplitAlongDim = _SplitAlongDim.apply

except ImportError:
    SplitAlongDim = None

try:
    from transformer_engine.pytorch.cpu_offload import (
        get_cpu_offload_context as _get_cpu_offload_context,
    )

    def get_cpu_offload_context(
        enabled,
        num_layers,
        model_layers,
        activation_offloading,
        weight_offloading,
        double_buffering,
        retain_pinned_cpu_buffers,
    ):
        """Get CPU offload context and sync function."""
        if is_te_min_version("2.10.0"):
            # TE 2.10+ supports retain_pinned_cpu_buffers
            context, sync_func = _get_cpu_offload_context(
                enabled,
                num_layers,
                model_layers,
                activation_offloading,
                weight_offloading,
                double_buffering,
                retain_pinned_cpu_buffers=retain_pinned_cpu_buffers,
            )
        elif is_te_min_version("2.5.0"):
            # TE 2.5-2.9 supports double_buffering but not retain_pinned_cpu_buffers
            context, sync_func = _get_cpu_offload_context(
                enabled,
                num_layers,
                model_layers,
                activation_offloading,
                weight_offloading,
                double_buffering,
            )
        elif is_te_min_version("1.10.0.dev0"):
            context, sync_func = _get_cpu_offload_context(
                enabled, num_layers, model_layers, activation_offloading, weight_offloading
            )
        else:
            context, sync_func = _get_cpu_offload_context(
                enabled, num_layers, activation_offloading, weight_offloading
            )

        return context, sync_func

except ImportError:
    get_cpu_offload_context = None  # type: ignore[assignment, misc]

try:
    if HAVE_TE and is_te_min_version("2.3.0"):
        from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb
    else:
        from transformer_engine.pytorch.attention import apply_rotary_pos_emb

    def fused_apply_rotary_pos_emb(
        t: torch.Tensor,
        freqs: torch.Tensor,
        transpose_output_memory: bool = False,
        interleaved: bool = False,
    ) -> torch.Tensor:
        """Apply rotary positional embedding to input tensor T in `sbhd` format."""
        if transpose_output_memory:
            warnings.warn(
                "transpose_output_memory is not supported by TE's fused RoPE and will be ignored."
            )
        if is_te_min_version("2.3.0"):
            return apply_rotary_pos_emb(
                t, freqs, tensor_format="sbhd", interleaved=interleaved, fused=True
            )
        else:
            if interleaved:
                raise ValueError("Only TE >= 2.3.0 supports interleaved fused RoPE.")

            return apply_rotary_pos_emb(t, freqs, tensor_format="sbhd", fused=True)

    def fused_apply_rotary_pos_emb_thd(
        t: torch.Tensor,
        cu_seqlens: torch.Tensor,
        freqs: torch.Tensor,
        cp_size: int = 1,
        cp_rank: int = 0,
        interleaved: bool = False,
    ) -> torch.Tensor:
        """
        Apply rotary positional embedding to input tensor T in `thd` format with CP support.
        """
        if interleaved:
            assert is_te_min_version("2.3.0"), "Only TE >= 2.3.0 supports interleaved fused RoPE."

        if is_te_min_version("2.3.0", check_equality=True):
            return apply_rotary_pos_emb(
                t,
                freqs,
                tensor_format="thd",
                fused=True,
                cu_seqlens=cu_seqlens,
                cp_size=cp_size,
                cp_rank=cp_rank,
                interleaved=interleaved,
            )
        elif is_te_min_version("1.12.0", check_equality=True):
            return apply_rotary_pos_emb(
                t,
                freqs,
                tensor_format="thd",
                fused=True,
                cu_seqlens=cu_seqlens,
                cp_size=cp_size,
                cp_rank=cp_rank,
            )
        else:
            assert cp_size == 1, "Only TE >= 1.12 supports RoPE fusion for THD format with CP."
            return apply_rotary_pos_emb(
                t, freqs, tensor_format="thd", fused=True, cu_seqlens=cu_seqlens
            )

except ImportError:
    pass

try:
    from transformer_engine.pytorch import Fp8Padding, Fp8Unpadding  # pylint: disable=unused-import

except ImportError:
    Fp8Padding = None
    Fp8Unpadding = None

try:
    from transformer_engine.pytorch.permutation import (
        moe_permute,
        moe_permute_with_probs,
        moe_sort_chunks_by_index,
        moe_sort_chunks_by_index_with_probs,
        moe_unpermute,
    )

    fused_permute = moe_permute
    fused_permute_with_probs = moe_permute_with_probs
    fused_sort_chunks_by_index = moe_sort_chunks_by_index
    fused_sort_chunks_by_index_with_probs = moe_sort_chunks_by_index_with_probs
    fused_unpermute = moe_unpermute

except ImportError:
    fused_permute = None
    fused_permute_with_probs = None
    fused_sort_chunks_by_index = None
    fused_sort_chunks_by_index_with_probs = None
    fused_unpermute = None

try:
    from transformer_engine.pytorch.permutation import moe_permute_and_pad_with_probs

    fused_permute_and_pad_with_probs = moe_permute_and_pad_with_probs

except ImportError:
    fused_permute_and_pad_with_probs = None

try:
    from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

    _TE_SUPPORTS_CG_CAPTURABLE = is_te_min_version("2.7.0")
    current_te_version = get_te_version()

    def te_parallel_cross_entropy(
        logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group: torch.distributed.ProcessGroup,
        is_cg_capturable: bool = False,
    ):
        """Wrapper function for TE's Cross Entropy Loss kernel"""
        if _TE_SUPPORTS_CG_CAPTURABLE:
            # According to TE CrossEntropyFunction, ignore_idx defaults to -100
            return parallel_cross_entropy(
                logits, labels, 0.0, False, tp_group, -100, is_cg_capturable
            )
        else:
            return parallel_cross_entropy(logits, labels, 0.0, False, tp_group)

except ImportError:
    te_parallel_cross_entropy = None  # type: ignore[assignment, misc]

try:
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    try:
        from transformer_engine.pytorch.module.base import get_workspace

        _get_workspace = get_workspace
    except ImportError:
        _get_workspace = None

    def te_general_gemm(
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: Optional[torch.dtype] = None,
        layout: str = "TN",
        out: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        grad: bool = False,
    ) -> List[torch.Tensor]:
        """
        Wrapper for TE's general_gemm function.
        It supports fp32, bf16, fp16, and fp8 GEMMs with TN, NN, and NT layouts.
        The output dtype can be specified by `out_dtype`.
        Note: not all combinations of these settings are supported. If not supported,
        cublaslt will throw an error.
        """
        kwargs = dict(
            out_dtype=out_dtype,
            quantization_params=None,
            gelu=None,
            gelu_in=None,
            accumulate=False,
            layout=layout,
            out=out,
            bias=bias,
            use_split_accumulator=False,
            grad=grad,
            ub=None,
            ub_type=None,
            extra_output=None,
            bulk_overlap=False,
        )
        if _get_workspace is not None:
            kwargs["workspace"] = _get_workspace()
        return general_gemm(A, B, **kwargs)

except ImportError:
    te_general_gemm = None  # type: ignore[assignment, misc]


if HAVE_TE and is_te_min_version("2.7.0.dev"):
    from transformer_engine.pytorch.router import (  # pylint: disable=unused-import
        fused_compute_score_for_moe_aux_loss,
        fused_moe_aux_loss,
        fused_topk_with_score_function,
    )

else:
    fused_topk_with_score_function = None
    fused_compute_score_for_moe_aux_loss = None
    fused_moe_aux_loss = None


def set_save_original_input(module):
    """
    Set the module to save the original input tensors.

    Some transformer-engine modules would save the quantized tensors by default in fp8 training.
    This method is used to set these modules to save the original input tensors directly.

    This can save the memory usage in some FP8 training scenarios, such as the attn linear_proj and
    the shared experts.
    The output-discarding recompute method also relies on this.
    """
    if hasattr(module, 'save_original_input'):
        module.save_original_input = True
    else:
        raise ValueError(
            "set_save_original_input is only needed on transformer-engine modules that save "
            "quantized tensors by default. It needs transformer-engine>=2.6.0dev0."
        )


try:
    # pylint: disable=unused-import
    from transformer_engine.pytorch import cpu_offload_v1 as cpu_offload
except ImportError:
    try:
        from transformer_engine.pytorch import cpu_offload
    except ImportError:
        cpu_offload = None
try:
    # pylint: disable=unused-import
    from transformer_engine.pytorch.float8_tensor import Float8Tensor
except ImportError:
    Float8Tensor = None


def get_thd_partitioned_indices(cu_seqlens, total_tokens, cp_size, cp_rank):
    """Get partitioned indices for THD format data in context parallel.

    Args:
        cu_seqlens: Cumulative sequence lengths tensor.
        total_tokens: Total number of tokens.
        cp_size: Context parallel world size.
        cp_rank: Context parallel rank.

    Returns:
        Partitioned indices tensor.
    """
    assert is_te_min_version("1.10.0"), (
        "Please update Transformer Engine to >= 1.10 to use "
        "Context Parallel with THD format data"
    )
    import transformer_engine_torch as tex

    return tex.thd_get_partitioned_indices(cu_seqlens, total_tokens, cp_size, cp_rank)
