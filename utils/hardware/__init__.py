from .backends import (
    AMP_DTYPE_ENV_VAR,
    BACKEND_ENV_VAR,
    HardwareBackend,
    amp_requires_grad_scaler,
    autocast_context,
    autocast_dtype,
    configure_stdio,
    describe_device,
    get_device_info,
    is_accelerator,
    is_amd,
    make_grad_scaler,
    no_autocast,
    normalize_device,
    reset_device_cache,
    resolve_amp_dtype,
    supports_amp,
    supports_bfloat16,
    supports_grad_scaler,
)
from .hardware_utils import (
    configure_miopen_cache,
    enable_fast_math,
    load_state_dict,
    print_backend_banner,
    print_model_params,
    resolve_amp_settings,
    save_state_dict,
    scaler_is_enabled,
    select_device,
)

# Every entry point imports this package before any model is built, so this is
# the earliest safe place to make console output robust on legacy Windows code
# pages and to give MIOpen a cache it can actually write to. MIOpen reads its
# cache variables when it is first used, which is well after this import.
configure_stdio()
configure_miopen_cache()

__all__ = [
    "AMP_DTYPE_ENV_VAR",
    "BACKEND_ENV_VAR",
    "HardwareBackend",
    "amp_requires_grad_scaler",
    "autocast_context",
    "autocast_dtype",
    "configure_miopen_cache",
    "configure_stdio",
    "describe_device",
    "enable_fast_math",
    "get_device_info",
    "is_accelerator",
    "is_amd",
    "load_state_dict",
    "make_grad_scaler",
    "no_autocast",
    "normalize_device",
    "print_backend_banner",
    "print_model_params",
    "reset_device_cache",
    "resolve_amp_dtype",
    "resolve_amp_settings",
    "save_state_dict",
    "scaler_is_enabled",
    "select_device",
    "supports_amp",
    "supports_bfloat16",
    "supports_grad_scaler",
]
