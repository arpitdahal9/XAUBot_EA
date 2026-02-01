"""
News Filter for Backtest Avoidance

Uses CSV calendar for reliable news event filtering.
O(log n) lookup using bisect for performance.

ONLY used in backtesting - live trading pauses manually.
"""

import os
import csv
import bisect
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NewsEvent:
    timestamp: datetime
    event_type: str
    currency: str
    importance: str
    description: str = ""


class NewsFilter:
    """
    Filter trades around major news events.
    
    Uses CSV calendar with O(log n) bisect lookup.
    Only active in backtesting.
    """
    
    def __init__(self, avoidance_minutes: int = 30, calendar_path: Optional[str] = None):
        self.avoidance_minutes = avoidance_minutes
        self.calendar_path = calendar_path
        self._events: List[NewsEvent] = []
        self._timestamps: List[datetime] = []  # Sorted for bisect
        self._loaded = False
        
        if calendar_path:
            self._load_calendar(calendar_path)
    
    def _load_calendar(self, path: str) -> bool:
        """Load news calendar from CSV file."""
        if not os.path.exists(path):
            logger.warning(f"News calendar not found: {path}. News filter will be disabled.")
            return False
        
        try:
            events = []
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ts_str = row.get('timestamp_utc') or row.get('timestamp', '')
                        ts = datetime.fromisoformat(ts_str.replace('Z', '').replace('T', ' '))
                        event = NewsEvent(
                            timestamp=ts,
                            event_type=row.get('event_type', 'UNKNOWN'),
                            currency=row.get('currency', 'USD'),
                            importance=row.get('importance', 'HIGH'),
                            description=row.get('description', '')
                        )
                        events.append(event)
                    except (ValueError, KeyError) as e:
                        continue
            
            # Sort by timestamp
            events.sort(key=lambda x: x.timestamp)
            self._events = events
            self._timestamps = [e.timestamp for e in events]
            self._loaded = True
            
            logger.info(f"Loaded {len(events)} news events from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load news calendar: {e}")
            return False
    
    def should_skip_trade(self, timestamp: datetime) -> bool:
        """
        Check if trade should be skipped due to nearby news.
        
        Uses bisect for O(log n) lookup.
        """
        if not self._loaded or not self._timestamps:
            return False
        
        window = timedelta(minutes=self.avoidance_minutes)
        
        # Find insertion point
        idx = bisect.bisect_left(self._timestamps, timestamp)
        
        # Check event at idx (next event after timestamp)
        if idx < len(self._timestamps):
            if abs((self._timestamps[idx] - timestamp).total_seconds()) <= window.total_seconds():
                return True
        
        # Check event at idx-1 (previous event before timestamp)
        if idx > 0:
            if abs((self._timestamps[idx - 1] - timestamp).total_seconds()) <= window.total_seconds():
                return True
        
        return False
    
    def get_nearest_event(self, timestamp: datetime) -> Optional[Tuple[NewsEvent, float]]:
        """Get nearest news event and distance in minutes."""
        if not self._loaded or not self._timestamps:
            return None
        
        idx = bisect.bisect_left(self._timestamps, timestamp)
        
        nearest = None
        min_dist = float('inf')
        
        # Check surrounding events
        for i in [idx - 1, idx]:
            if 0 <= i < len(self._events):
                dist = abs((self._timestamps[i] - timestamp).total_seconds()) / 60
                if dist < min_dist:
                    min_dist = dist
                    nearest = self._events[i]
        
        if nearest:
            return (nearest, min_dist)
        return None
    
    def get_events_in_range(self, start: datetime, end: datetime) -> List[NewsEvent]:
        """Get all events in date range using bisect."""
        if not self._loaded:
            return []
        
        start_idx = bisect.bisect_left(self._timestamps, start)
        end_idx = bisect.bisect_right(self._timestamps, end)
        
        return self._events[start_idx:end_idx]
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    @property
    def event_count(self) -> int:
        return len(self._events)
