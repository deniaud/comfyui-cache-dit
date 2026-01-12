"""
CacheDiT API Compatibility Layer


This module provides a standard CacheDiT API interface while maintaining perfect compatibility with the ComfyUI ModelPatcher architecture.
It supports all API invocation methods from the original CacheDiT:


```python
import cache_dit
cache_dit.enable_cache(model, skip_interval=2, warmup_steps=3)
cache_dit.disable_cache(model)
cache_dit.summary(model)
```


Internally it uses an enhanced caching engine, offering more flexible configuration options and detailed performance statistics.
"""


from .cache_engine import global_cache, CacheStrategy
from typing import Optional, Dict, Any, Union
import warnings



def enable_cache(model, **cache_options) -> None:
    """
    Enable CacheDiT cache acceleration for a model
    
    This is the standard CacheDiT API entry point and is fully compatible with the original API.
    It supports multiple caching strategies and configuration options.
    
    Args:
        model: Model object (supports ComfyUI ModelPatcher and other model types)
        **cache_options: Cache configuration options
            - skip_interval (int): Step-skipping interval, default is 2 (skip every other step)
            - warmup_steps (int): Number of warmup steps, default is 3
            - strategy (str): Cache strategy, one of 'fixed', 'dynamic', 'adaptive'
            - noise_scale (float): Noise scaling factor, default is 0.001
            - enable_stats (bool): Whether to enable detailed statistics, default is True
            - debug (bool): Whether to enable debug output, default is False
    
    Example:
        ```python
        import cache_dit
        
        # Basic usage
        cache_dit.enable_cache(model)
        
        # Custom configuration
        cache_dit.enable_cache(model, 
                              skip_interval=3, 
                              warmup_steps=5,
                              strategy='adaptive')
        ```
    """
    print(f"\n🚀 Enable CacheDiT acceleration (API compatibility mode)")
    
    # Set default configuration
    default_options = {
        'skip_interval': 2,
        'warmup_steps': 3,
        'strategy': 'fixed',
        'noise_scale': 0.001,
        'enable_stats': True,
        'debug': False
    }
    
    # Merge user configuration
    config = {**default_options, **cache_options}
    
    # Create cache strategy
    strategy = CacheStrategy(
        skip_interval=config['skip_interval'],
        warmup_steps=config['warmup_steps'],
        strategy_type=config['strategy'],
        noise_scale=config['noise_scale'],
        enable_stats=config['enable_stats'],
        debug=config['debug']
    )
    
    print(f"   Configuration: {config}")
    
    # Apply caching to the model
    global_cache.enable_cache(model, strategy)
    
    print("✓ CacheDiT cache enabled")



def disable_cache(model) -> None:
    """
    Disable CacheDiT caching for a model
    
    Restore the model's original behavior and remove all cache-related modifications.
    This is the standard CacheDiT API interface.
    
    Args:
        model: The model object that previously had caching applied
        
    Example:
        ```python
        import cache_dit
        
        # Disable cache
        cache_dit.disable_cache(model)
        ```
    """
    print(f"\n🛑 Disable CacheDiT cache (API compatibility mode)")
    
    # Disable cache via the global cache manager
    global_cache.disable_cache(model)
    
    print("✓ CacheDiT cache disabled")



def summary(model) -> str:
    """
    Get a cache statistics summary for the model
    
    Returns detailed cache performance statistics, including hit rate, speedup ratio, etc.
    This is the standard CacheDiT API interface.
    
    Args:
        model: Model object
        
    Returns:
        str: Formatted statistics string
        
    Example:
        ```python
        import cache_dit
        
        # Get statistics
        stats = cache_dit.summary(model)
        print(stats)
        ```
    """
    print(f"\n📊 Get CacheDiT statistics summary (API compatibility mode)")
    
    # Get global statistics
    stats = global_cache.get_detailed_stats()
    
    return stats



def set_global_config(**config) -> None:
    """
    Set global cache configuration
    
    Affects the default behavior of all subsequent enable_cache calls.
    This is an extended API that provides more flexible global configuration management.
    
    Args:
        **config: Global configuration options
            - default_skip_interval (int): Default step-skipping interval
            - default_warmup_steps (int): Default number of warmup steps
            - default_strategy (str): Default cache strategy
            - global_debug (bool): Global debug mode
            
    Example:
        ```python
        import cache_dit
        
        # Set global configuration
        cache_dit.set_global_config(
            default_skip_interval=3,
            default_strategy='adaptive',
            global_debug=True
        )
        ```
    """
    print(f"\n⚙️ Set CacheDiT global configuration")
    print(f"   Options: {config}")
    
    global_cache.set_global_config(config)
    
    print("✓ Global configuration updated")



def get_global_stats() -> Dict[str, Any]:
    """
    Get global cache statistics
    
    Returns a detailed dictionary containing cache statistics for all models.
    This is an extended API for advanced monitoring and debugging.
    
    Returns:
        Dict[str, Any]: Dictionary containing detailed statistics
        
    Example:
        ```python
        import cache_dit
        
        # Get global statistics
        stats = cache_dit.get_global_stats()
        print(f"Total cache hits: {stats['total_cache_hits']}")
        ```
    """
    return global_cache.get_global_stats()



def reset_cache_stats() -> None:
    """
    Reset all cache statistics
    
    Clears all counters and performance metrics to restart statistics collection.
    This is an extended API for debugging and testing.
    
    Example:
        ```python
        import cache_dit
        
        # Reset statistics
        cache_dit.reset_cache_stats()
        ```
    """
    print(f"\n🔄 Reset CacheDiT statistics")
    
    global_cache.reset_stats()
    
    print("✓ Statistics reset")



# Compatibility aliases - support different import styles
enable = enable_cache
disable = disable_cache
stats = summary


# Version information
__version__ = "1.0.0"
__api_version__ = "compatible"


# Exported public interface
__all__ = [
    'enable_cache',
    'disable_cache', 
    'summary',
    'set_global_config',
    'get_global_stats',
    'reset_cache_stats',
    # Aliases
    'enable',
    'disable',
    'stats'
]