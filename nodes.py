"""
ComfyUI Cache Acceleration Nodes


This file defines custom nodes for ComfyUI. Users can apply cache acceleration in workflows via these nodes.
Based on the enhanced caching engine, it supports multiple caching strategies and configuration options.


New features:
- Cache configuration node: supports multiple strategies and parameter tuning
- Cache control node: dynamically enable/disable caching
- Enhanced statistics node: provides more detailed performance information
"""


from .cache_engine import patch_model_simple, get_simple_stats, global_cache, CacheStrategy



class CacheDitAccelerateNode:
    """
    CacheDit acceleration node
    
    Applies cache acceleration to a ComfyUI model to achieve \(2x+\) inference speedup
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }


    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("Accelerated model",)
    FUNCTION = "accelerate_model"
    CATEGORY = "CacheDit"


    def accelerate_model(self, model):
        """
        Apply cache acceleration to the model
        
        Args:
            model: Input ComfyUI model
            
        Returns:
            tuple: (accelerated model,)
        """
        print("\n🚀 Applying CacheDit acceleration...")
        
        # Apply the cache patch
        accelerated_model = patch_model_simple(model)
        
        print("✓ CacheDit acceleration applied")
        return (accelerated_model,)



class CacheDitStatsNode:
    """
    CacheDit statistics node
    
    Displays cache statistics, including hit rate and expected speedup
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("*",),  # Accept any type as a trigger
            }
        }


    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Statistics",)
    FUNCTION = "get_stats"
    CATEGORY = "CacheDit"


    def get_stats(self, trigger):
        """
        Get cache statistics
        
        Args:
            trigger: Trigger (any value)
            
        Returns:
            tuple: (statistics string,)
        """
        stats = get_simple_stats()
        return (stats,)



class CacheDitConfigNode:
    """
    CacheDit configuration node
    
    Provides advanced cache configuration options, supporting multiple caching strategies and parameter tuning
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "strategy": (["fixed", "dynamic", "adaptive"], {"default": "fixed"}),
                "skip_interval": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "warmup_steps": ("INT", {"default": 3, "min": 0, "max": 20, "step": 1}),
            },
            "optional": {
                "noise_scale": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 0.1, "step": 0.001}),
                "enable_debug": ("BOOLEAN", {"default": False}),
            }
        }


    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("Configured model",)
    FUNCTION = "configure_cache"
    CATEGORY = "CacheDit"


    def configure_cache(self, model, strategy, skip_interval, warmup_steps, 
                       noise_scale=0.001, enable_debug=False):
        """
        Configure and apply cache acceleration to the model
        
        Args:
            model: Input ComfyUI model
            strategy: Cache strategy type
            skip_interval: Step-skipping interval
            warmup_steps: Warmup steps
            noise_scale: Noise scaling factor
            enable_debug: Whether to enable debug output
            
        Returns:
            tuple: (configured model,)
        """
        print(f"\n🔧 Configuring CacheDit acceleration...")
        print(f"   Strategy: {strategy}")
        print(f"   Skip interval: {skip_interval}")
        print(f"   Warmup steps: {warmup_steps}")
        print(f"   Noise scale: {noise_scale}")
        print(f"   Debug mode: {enable_debug}")
        
        # Create cache strategy
        cache_strategy = CacheStrategy(
            skip_interval=skip_interval,
            warmup_steps=warmup_steps,
            strategy_type=strategy,
            noise_scale=noise_scale,
            enable_stats=True,
            debug=enable_debug
        )
        
        # Apply cache configuration
        configured_model = global_cache.enable_cache(model, cache_strategy)
        
        print("✓ CacheDit cache configuration applied")
        return (configured_model,)



class CacheDitControlNode:
    """
    CacheDit control node
    
    Dynamically enable or disable the model's caching feature
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable_cache": ("BOOLEAN", {"default": True}),
            }
        }


    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("Controlled model",)
    FUNCTION = "control_cache"
    CATEGORY = "CacheDit"


    def control_cache(self, model, enable_cache):
        """
        Control the cache enabled state for the model
        
        Args:
            model: Input ComfyUI model
            enable_cache: Whether to enable cache
            
        Returns:
            tuple: (controlled model,)
        """
        print(f"\n🎛 Controlling CacheDit cache: {'Enable' if enable_cache else 'Disable'}")
        
        if enable_cache:
            # Enable cache (use default strategy)
            controlled_model = global_cache.enable_cache(model, CacheStrategy())
        else:
            # Disable cache
            global_cache.disable_cache(model)
            controlled_model = model
        
        print(f"✓ CacheDit cache {'enabled' if enable_cache else 'disabled'}")
        return (controlled_model,)



class CacheDitDetailedStatsNode:
    """
    CacheDit detailed statistics node
    
    Displays detailed cache statistics, including multi-model stats
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("*",),  # Accept any type as a trigger
            },
            "optional": {
                "show_model_details": ("BOOLEAN", {"default": True}),
            }
        }


    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Detailed statistics",)
    FUNCTION = "get_detailed_stats"
    CATEGORY = "CacheDit"


    def get_detailed_stats(self, trigger, show_model_details=True):
        """
        Get detailed cache statistics
        
        Args:
            trigger: Trigger (any value)
            show_model_details: Whether to show model details
            
        Returns:
            tuple: (detailed statistics string,)
        """
        if show_model_details:
            stats = global_cache.get_detailed_stats()
        else:
            stats = global_cache.get_stats()
        return (stats,)


class CacheDitAccelerateWanVideoWrapperNode:
    """
    CacheDit acceleration node for WanVideoWrapper

    Applies cache acceleration to a WanVideo model to achieve \(2x+\) inference speedup
    during video generation. This node is specifically designed to work with WanVideo models
    and integrates seamlessly with WanVideoSampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),  # Changed to accept WANVIDEOMODEL
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)  # Changed to return WANVIDEOMODEL
    RETURN_NAMES = ("Accelerated model",)
    FUNCTION = "accelerate_model"
    CATEGORY = "CacheDit/WanVideoWrapper"

    def accelerate_model(self, model):
        """
        Apply cache acceleration to the WanVideo model

        Args:
            model: Input WanVideo model

        Returns:
            tuple: (accelerated model,)
        """
        print("\n🚀 Applying CacheDit acceleration to WanVideo model...")

        # Apply the cache patch to the WanVideo model
        # The WanVideo model has a diffusion_model attribute that contains the actual transformer
        # We need to patch the underlying diffusion model
        accelerated_model = patch_model_simple(model)

        print("✓ CacheDit acceleration applied to WanVideo model")
        return (accelerated_model,)


class CacheDitConfigWanVideoWrapperNode:
    """
    CacheDit configuration node for WanVideoWrapper

    Provides advanced cache configuration options for WanVideo models, supporting multiple caching strategies and parameter tuning
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "strategy": (["fixed", "dynamic", "adaptive"], {"default": "fixed"}),
                "skip_interval": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "warmup_steps": ("INT", {"default": 3, "min": 0, "max": 20, "step": 1}),
            },
            "optional": {
                "noise_scale": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 0.1, "step": 0.001}),
                "enable_debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("Configured model",)
    FUNCTION = "configure_cache"
    CATEGORY = "CacheDit/WanVideoWrapper"

    def configure_cache(self, model, strategy, skip_interval, warmup_steps,
                       noise_scale=0.001, enable_debug=False):
        """
        Configure and apply cache acceleration to the WanVideo model

        Args:
            model: Input WanVideo model
            strategy: Cache strategy type
            skip_interval: Step-skipping interval
            warmup_steps: Warmup steps
            noise_scale: Noise scaling factor
            enable_debug: Whether to enable debug output

        Returns:
            tuple: (configured model,)
        """
        print(f"\n🔧 Configuring CacheDit acceleration for WanVideo model...")
        print(f"   Strategy: {strategy}")
        print(f"   Skip interval: {skip_interval}")
        print(f"   Warmup steps: {warmup_steps}")
        print(f"   Noise scale: {noise_scale}")
        print(f"   Debug mode: {enable_debug}")

        # Create cache strategy
        cache_strategy = CacheStrategy(
            skip_interval=skip_interval,
            warmup_steps=warmup_steps,
            strategy_type=strategy,
            noise_scale=noise_scale,
            enable_stats=True,
            debug=enable_debug
        )

        # Apply cache configuration to the WanVideo model
        configured_model = global_cache.enable_cache(model, cache_strategy)

        print("✓ CacheDit cache configuration applied to WanVideo model")
        return (configured_model,)


class CacheDitControlWanVideoWrapperNode:
    """
    CacheDit control node for WanVideoWrapper

    Dynamically enable or disable the WanVideo model's caching feature
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "enable_cache": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("Controlled model",)
    FUNCTION = "control_cache"
    CATEGORY = "CacheDit/WanVideoWrapper"

    def control_cache(self, model, enable_cache):
        """
        Control the cache enabled state for the WanVideo model

        Args:
            model: Input WanVideo model
            enable_cache: Whether to enable cache

        Returns:
            tuple: (controlled model,)
        """
        print(f"\n🎛 Controlling CacheDit cache for WanVideo model: {'Enable' if enable_cache else 'Disable'}")

        if enable_cache:
            # Enable cache (use default strategy)
            controlled_model = global_cache.enable_cache(model, CacheStrategy())
        else:
            # Disable cache
            global_cache.disable_cache(model)
            controlled_model = model

        print(f"✓ CacheDit cache for WanVideo model {'enabled' if enable_cache else 'disabled'}")
        return (controlled_model,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    # Original nodes (kept for backward compatibility)
    "CacheDitAccelerate": CacheDitAccelerateNode,
    "CacheDitStats": CacheDitStatsNode,
    
    # New enhanced nodes
    "CacheDitConfig": CacheDitConfigNode,
    "CacheDitControl": CacheDitControlNode,
    "CacheDitDetailedStats": CacheDitDetailedStatsNode,
    "CacheDitAccelerateWanVideoWrapper": CacheDitAccelerateWanVideoWrapperNode,
    "CacheDitConfigWanVideoWrapper": CacheDitConfigWanVideoWrapperNode,
    "CacheDitControlWanVideoWrapper": CacheDitControlWanVideoWrapperNode,
}


# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    # Original nodes
    "CacheDitAccelerate": "CacheDit Model Acceleration",
    "CacheDitStats": "CacheDit Statistics",
    
    # New nodes
    "CacheDitConfig": "CacheDit Advanced Configuration",
    "CacheDitControl": "CacheDit Cache Control",
    "CacheDitDetailedStats": "CacheDit Detailed Statistics",
    "CacheDitAccelerateWanVideoWrapper": "CacheDit WanVideo Model Acceleration",
    "CacheDitConfigWanVideoWrapper": "CacheDit WanVideo Advanced Configuration",
    "CacheDitControlWanVideoWrapper": "CacheDit WanVideo Cache Control",
}