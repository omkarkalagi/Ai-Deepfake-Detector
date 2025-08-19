
import json
import os
from functools import lru_cache
from pathlib import Path

class OptimizedDataLoader:
    """Optimized data loading with caching."""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def load_gallery_data():
        """Load gallery data with caching."""
        try:
            gallery_path = Path('static/gallery/gallery_data.json')
            if gallery_path.exists():
                with open(gallery_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading gallery data: {e}")
        return {}
    
    @staticmethod
    @lru_cache(maxsize=128)
    def load_gallery_stats():
        """Load gallery stats with caching."""
        try:
            stats_path = Path('static/gallery/gallery_stats.json')
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading gallery stats: {e}")
        return {}
    
    @staticmethod
    def clear_cache():
        """Clear all cached data."""
        OptimizedDataLoader.load_gallery_data.cache_clear()
        OptimizedDataLoader.load_gallery_stats.cache_clear()
