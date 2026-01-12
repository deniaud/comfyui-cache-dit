Here is the English translation, keeping the original Markdown formatting and structure intact.

***

# ComfyUI CacheDit Plugin

This is a powerful plugin providing diffusion model inference acceleration for **ComfyUI**, **fully compatible with the standard CacheDiT API**, while offering rich ComfyUI node support.

🎉 **New Version Features**:
- ✅ **Fully Compatible with Standard CacheDiT API** - Supports standard interfaces like `cache_dit.enable_cache()`
- ✅ **Multiple Caching Strategies** - Choose from Fixed, Dynamic, and Adaptive strategies
- ✅ **Flexible Configuration** - Supports adjustment of skip intervals, warmup steps, noise scaling, and other parameters
- ✅ **Detailed Statistics** - Provides multi-level performance statistics and monitoring
- ✅ **Backward Compatibility** - Maintains all original functionality

***

## ✨ Features

### Core Functions
- **Simple and Direct**: Out-of-the-box, no extra configuration required
- **Significant Acceleration**: Achieves nearly 2x inference speedup on models like FLUX
- **Plug and Play**: Use it in workflows just like a standard node
- **Easy Debugging**: Includes basic performance statistical information

### New Functions
- **Standard API Compatibility**: Supports all API calls of the original CacheDiT
- **Multiple Caching Strategies**: Fixed skipping, dynamic skipping, adaptive strategy
- **Advanced Configuration**: Configurable skip intervals, warmup steps, noise scaling
- **Multi-Model Support**: Manage cache states for multiple models simultaneously
- **Detailed Monitoring**: Model-level statistics and performance analysis

***

## 📦 Installation

1. Place the plugin folder into the ComfyUI `custom_nodes` directory
2. Restart ComfyUI

***

## 🔧 Usage

### Method 1: Standard CacheDiT API (Recommended)

**Fully compatible with the original CacheDiT API**, you can directly replace existing code:

```python
# Import CacheDiT API
import cache_dit

# Basic Usage
cache_dit.enable_cache(model)

# Advanced Usage
cache_dit.enable_cache(model, 
                      skip_interval=3,      # Skip 1 step every 3 steps
                      warmup_steps=5,       # Warmup for the first 5 steps
                      strategy='adaptive',  # Adaptive strategy
                      noise_scale=0.002)    # Noise scaling

# Get Statistics
stats = cache_dit.summary(model)
print(stats)

# Disable Cache
cache_dit.disable_cache(model)
```

### Method 2: ComfyUI Nodes

#### Basic Node (Backward Compatible)
1. Add the **`CacheDit Model Acceleration`** node to your workflow
2. Connect the model to this node
3. Use the accelerated model for inference

#### Advanced Nodes (New Features)
1. **`CacheDit Advanced Config`** - Supports multiple strategies and parameter adjustments
2. **`CacheDit Cache Control`** - Dynamically enable/disable cache
3. **`CacheDit Detailed Stats`** - View detailed performance analysis

Recommended Workflow:
```
Model Loader → CacheDit Advanced Config → Inference Node → CacheDit Detailed Stats
```

***

## 📊 Caching Strategies

### 1. Fixed Strategy (Fixed)
- Skips calculation at fixed intervals
- Suitable for stable inference scenarios
- Parameter: `skip_interval`

### 2. Dynamic Strategy (Dynamic)
- Skip frequency increases as steps increase
- Suitable for long sequence inference
- Automatically optimizes performance

### 3. Adaptive Strategy (Adaptive)
- Automatically adjusts based on performance monitoring
- Smartly optimizes caching strategy
- Best performance

***

## ⚡ Performance

- **FLUX Model**: Acceleration up to approx. 2x
- **Cache Hit Rate**: Theoretically approx. 50% (Fixed strategy)
- **Quality Impact**: Almost indistinguishable to the naked eye
- **Multi-Model Support**: Accelerates multiple models simultaneously

***

## 🔧 Advanced Configuration

### API Method
```python
import cache_dit

# Set Global Configuration
cache_dit.set_global_config(
    default_skip_interval=3,
    default_strategy='adaptive',
    global_debug=True
)

# Get Global Stats
stats = cache_dit.get_global_stats()
print(f"Total Cache Hits: {stats['total_cache_hits']}")

# Reset Statistics
cache_dit.reset_cache_stats()
```

### Node Method
Using the **`CacheDit Advanced Config`** node:
- Strategy Selection: fixed/dynamic/adaptive
- Skip Interval: 1-10
- Warmup Steps: 0-20
- Noise Scaling: 0.0-0.1
- Debug Mode: On/Off

***

## 📊 How It Works

### Basic Principle
- Runs normally for the first few steps (Warmup phase)
- Afterwards, it skips parts of the calculation based on the strategy, directly reusing previous results
- Adds slight noise to the reused results to avoid obvious artifacts in the image

This approach leverages the characteristic that adjacent steps in diffusion models have similar results, thus saving significant computation.

### Strategy Details
1. **Fixed Strategy**: Skips 1 step every N steps, simple and reliable
2. **Dynamic Strategy**: Dynamically adjusts skip frequency as step count increases
3. **Adaptive Strategy**: Smartly optimizes skip decisions based on performance monitoring

***

## 🛠 Troubleshooting

If the node is not taking effect, try the following:
1. Check the debug logs output in the console
2. Confirm if the model type is compatible (currently mainly supports Transformer architectures)
3. Check statistics to confirm if the cache is actually being used
4. Use debug mode: `cache_dit.enable_cache(model, debug=True)`
5. View detailed stats: Use the `CacheDit Detailed Stats` node

### Common Issues
- **Poor Caching Effect**: Adjust skip interval and warmup steps
- **Image Quality Degradation**: Reduce noise scaling factor or increase warmup steps
- **Transformer Not Found**: Check model type compatibility

***

## 📚 More Resources

- **Usage Examples**: Check `examples.py` for detailed usage
- **API Documentation**: All functions have detailed docstrings
- **Debug Guide**: Enable debug mode to get detailed logs
- **Performance Testing**: Use statistical functions to monitor acceleration effects

***

## 🚧 Roadmap

- [x] ✅ Implement Standard CacheDiT API Compatibility
- [x] ✅ Support Multiple Caching Strategies
- [x] ✅ Add Detailed Statistics and Monitoring
- [x] ✅ Create Advanced Configuration Node
- [ ] 🔄 Add More Adaptive Strategy Algorithms
- [ ] 🔄 Support More Model Architectures
- [ ] 🔄 Performance Benchmark Suite

***

## 📄 License

Open source project, issues and PRs welcome.

***

Would you like me to create a Python script example demonstrating how to integrate this `cache_dit` API into a standard Diffusers pipeline?
